package ipvs

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	dto "github.com/prometheus/client_model/go"
	p2j "github.com/prometheus/prom2json"
	"google.golang.org/grpc"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	v1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsclientset "k8s.io/metrics/pkg/client/clientset/versioned"

	tfcoreframework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"
)

const (
	// parallelism for better CPU utilization,
	// using k8s.io/kubernetes/pkg/scheduler/internal/parallelize as a reference.
	parallelism = 16

	// MaxWeight is highest ipvs weight.(1 ~ 65535)
	MaxWeight = 100

	// Interval to get the temperature. (minutes)
	getTemperatureInterval = 15

	// Node label that TensorFlow Serving host.
	labelTensorflowHost = "tensorflow/host"
	// Node label that TensorFlow Serving port.
	labelTensorflowPort = "tensorflow/port"
	// Node label that TensorFlow Serving model name.
	labelTensorflowName = "tensorflow/name"
	// Node label that TensorFlow Serving model version. (optional label)
	labelTensorflowVersion = "tensorflow/version"
	// Node label that TensorFlow Serving model signature. (optional label)
	labelTensorflowSignature = "tensorflow/signature"

	// Node label that max ambient temperature.
	labelAmbientMax = "ambient/max"
	// Node label that min ambient temperature.
	labelAmbientMin = "ambient/min"
	// Node label that cpu1 max information.
	labelCPU1Max = "cpu1/max"
	// Node label that cpu1 min information.
	labelCPU1Min = "cpu1/min"
	// Node label that cpu2 max information.
	labelCPU2Max = "cpu2/max"
	// Node label that cpu2 min information.
	labelCPU2Min = "cpu2/min"

	// IPMI exporter port
	ipmiPort = "9290"
	// IPMI exporter protocol
	ipmiProtocol = "http://"
	// IPMI exporter metrics endpoint
	endpoint = "/metrics"
	// IPMI exporter temperature metrics key
	ipmiTemperatureKey = "ipmi_temperature_celsius"
	// IPMI exporter ambient temperature key
	ambientKey = "Ambient"
	// IPMI exporter CPU1 temperature key
	cpu1Key = "CPU1"
	// IPMI exporter CPU2 temperature key
	cpu2Key = "CPU2"
)

// nodeTemperatureInfo defines the input for calculate current power consumption.
type nodeTemperatureInfo struct {
	ambient          float32
	cpu1             float32
	cpu2             float32
	ambientTimestamp time.Time
}

// predictInput defines the input for power consumption predictions.
type predictInput struct {
	cpuUsage    float32
	ambientTemp float32
	cpu1Temp    float32
	cpu2Temp    float32
}

// cacheKey defines the key to use when caching power consumption predictions.
type cacheKey struct {
	server       string
	predictInput predictInput
}

// safeCache is safe to use concurrently cache.
type safeCache struct {
	cache map[cacheKey]float32
	sync.Mutex
}

// familyInfo defines IPMI exporter metrics objects.
type familyInfo struct {
	Name    string `json:"name"`
	Metrics []struct {
		Labels struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		}
		Value string `json:"value"`
	} `json:"metrics"`
}

// getInfomationsImpl implements the getInfomations interface.
type getInfomationsImpl struct{}

// getInfomations is the interface to get information.
type getInfomations interface {
	// Get node metrics from metrics server.
	getNodeMetrics(ctx context.Context, node string) (*v1beta1.NodeMetrics, error)
	// Get node sensor data from IPMI exporter.
	getFamilyInfo(url string) []*p2j.Family
	// Get prediction power consumption from TensorFlow Serving.
	predictPC(values predictInput, nodeInfo *v1.Node) (float32, error)
}

// Get node metrics from metrics server.
func (i getInfomationsImpl) getNodeMetrics(ctx context.Context, node string) (*v1beta1.NodeMetrics, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		klog.Warningf("%v : Cannot get InClusterConfig. Error : %v", node, err)
		return nil, err
	}

	mc, err := metricsclientset.NewForConfig(config)
	if err != nil {
		klog.V(5).Infof("%v : Cannot get metrics infomations because %v", node, err)
		return nil, err
	}

	nm := mc.MetricsV1beta1().NodeMetricses()
	nodemetrics, err := nm.Get(ctx, node, metav1.GetOptions{})
	klog.V(5).Infof("%v : Nodemetrics is %+v", node, nodemetrics)
	return nodemetrics, err
}

// Get node sensor data from IPMI exporter.
func (i getInfomationsImpl) getFamilyInfo(url string) []*p2j.Family {
	mfChan := make(chan *dto.MetricFamily, 1024)
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: false},
	}
	go func() {
		err := p2j.FetchMetricFamilies(url, mfChan, transport)
		if err != nil {
			klog.V(5).Infof("Cannot get metric from %v because %v", url, err)
		}
	}()

	result := []*p2j.Family{}
	for mf := range mfChan {
		result = append(result, p2j.NewFamily(mf))
	}
	return result
}

// Get prediction power consumption from TensorFlow Serving.
func (i getInfomationsImpl) predictPC(pi predictInput, nodeInfo *v1.Node) (float32, error) {
	values := []float32{pi.cpuUsage, pi.ambientTemp, pi.cpu1Temp, pi.cpu2Temp}

	servingAddress, hostOK := nodeInfo.Labels[labelTensorflowHost]
	servingPort, portOK := nodeInfo.Labels[labelTensorflowPort]
	name, nameOK := nodeInfo.Labels[labelTensorflowName]
	if !hostOK || !portOK || !nameOK {
		return -1, fmt.Errorf("Label is not defined. [%v: %v, %v: %v, %v: %v]",
			labelTensorflowHost, hostOK, labelTensorflowPort, portOK, labelTensorflowName, nameOK)
	}

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name: name,
		},
		Inputs: map[string]*tfcoreframework.TensorProto{
			"inputs": {
				Dtype: tfcoreframework.DataType_DT_FLOAT,
				TensorShape: &tfcoreframework.TensorShapeProto{
					Dim: []*tfcoreframework.TensorShapeProto_Dim{
						{
							Size: int64(1),
						},
						{
							Size: int64(4),
						},
					},
				},
				FloatVal: values,
			},
		},
	}
	if strVersion, ok := nodeInfo.Labels[labelTensorflowVersion]; ok {
		if intVersion, err := strconv.ParseInt(strVersion, 10, 64); err == nil {
			request.ModelSpec.Version = &google_protobuf.Int64Value{Value: intVersion}
		} else {
			klog.Warningf("Convert to int64 failed [%v : %v]", labelTensorflowVersion, strVersion)
		}
	}
	if signature, ok := nodeInfo.Labels[labelTensorflowSignature]; ok {
		request.ModelSpec.SignatureName = signature
	}

	conn, err := grpc.Dial(servingAddress+":"+servingPort, grpc.WithInsecure())
	if err != nil {
		return -1, err
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)

	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		return -1, err
	}
	return resp.Outputs["outputs"].FloatVal[0], nil
}

// Set predicted power consumption to the cache.
func (safeCache *safeCache) setPCCache(p predictInput, nodeName string, value float32) {
	safeCache.Lock()
	defer safeCache.Unlock()
	safeCache.cache[cacheKey{nodeName, p}] = value
}

// Get predicted power consumption from the cache.
func (safeCache *safeCache) getPCCache(p predictInput, nodeName string) (float32, bool) {
	safeCache.Lock()
	defer safeCache.Unlock()
	value, ok := safeCache.cache[cacheKey{nodeName, p}]
	return value, ok
}

// chunkSizeFor returns a chunk size for the given number of items to use for
// parallel work. The size aims to produce good CPU utilization.
// using k8s.io/kubernetes/pkg/scheduler/internal/parallelize as a reference.
func chunkSizeFor(n int) workqueue.Options {
	s := int(math.Sqrt(float64(n)))
	if r := n/parallelism + 1; s > r {
		s = r
	} else if s < 1 {
		s = 1
	}
	return workqueue.WithChunkSize(s)
}

// CalcWeight calculate endpoints weight
func (proxier *Proxier) CalcWeight(endpointlist []string) map[string]int {
	weight := make(map[string]int)
	klog.V(5).Infof("endpointlist: %v", endpointlist)
	if len(endpointlist) == 0 {
		return weight
	} else if len(endpointlist) == 1 {
		weight[endpointlist[0]] = 1
		return weight
	}

	lowest := int64(math.MaxInt64)
	for _, endpoint := range endpointlist {
		ip, _, err := net.SplitHostPort(endpoint)
		if err != nil {
			klog.Errorf("Failed to parse endpoint: %v, error: %v", endpoint, err)
			continue
		}
		tmpScore, ok := proxier.nodesScore[proxier.endpointsBelongNode[ip]]
		if !ok || tmpScore == -1 {
			continue
		}
		if tmpScore < lowest {
			lowest = tmpScore
		}
	}
	klog.V(5).Infof("Lowest score: %v", lowest)

	// Endpoint weight is larger as nodesScore is smaller.
	for _, endpoint := range endpointlist {
		// Endpoint weight set to 1, if the scores of all nodes could not be calculated.
		if lowest == int64(math.MaxInt64) {
			weight[endpoint] = 1
			continue
		}
		ip, _, err := net.SplitHostPort(endpoint)
		if err != nil {
			klog.Errorf("Failed to parse endpoint: %v, error: %v", endpoint, err)
			continue
		}
		tmpScore, ok := proxier.nodesScore[proxier.endpointsBelongNode[ip]]
		if !ok || tmpScore == -1 {
			weight[endpoint] = 0
			continue
		}
		weight[endpoint] = int(lowest * MaxWeight / tmpScore)
	}
	return weight
}

// Get standard temperature informations from the node's label.
func getStandardTemperature(nodeInfo *v1.Node, labelKeyMax string, labelKeyMin string) (float32, float32, error) {
	tempMax, maxOK := nodeInfo.Labels[labelKeyMax]
	tempMin, minOK := nodeInfo.Labels[labelKeyMin]
	if !maxOK || !minOK {
		klog.V(5).Infof("%v : Standard temperature information label is not defined. [%v : %v, %v : %v] ",
			nodeInfo.Name, labelKeyMax, maxOK, labelKeyMin, minOK)
		return -1, -1, errors.New("Standard temperature informations label is not defined")
	}

	maxValue, err := strconv.ParseFloat(tempMax, 32)
	if err != nil {
		klog.V(5).Infof("Convert to float64 failed. [%v : %v]", labelKeyMax, tempMax)
		return -1, -1, errors.New("Convert to float64 failed")
	}

	minValue, err := strconv.ParseFloat(tempMin, 32)
	if err != nil {
		klog.V(5).Infof("Convert to float64 failed. [%v : %v]", labelKeyMin, tempMin)
		return -1, -1, errors.New("Convert to float64 failed")
	}

	// The standard temperature informations max and min must not be the same because division by zero occurs in calcNormalizeTemperature.
	if (maxValue - minValue) == 0 {
		klog.V(5).Infof("%v : Do not set values of %v and %v the same", nodeInfo.Name, labelKeyMax, labelKeyMin)
		return -1, -1, errors.New("division by zero")
	}
	return float32(maxValue), float32(minValue), nil
}

// Get the node's internal IP for the IPMI exporter's address.
func getNodeInternalIP(node *v1.Node) (string, error) {
	for _, addres := range node.Status.Addresses {
		if addres.Type == v1.NodeInternalIP {
			return addres.Address, nil
		}
	}
	return "", errors.New("Cannot get node internalIP")
}

// Get node Ambient and CPU1 and CPU2 temperatures used to predict power consumption.
func (proxier *Proxier) getNodeTemperature(nodeAddress string) ([]float32, error) {
	var url string = strings.Join([]string{ipmiProtocol, nodeAddress, ":", ipmiPort, endpoint}, "")
	result := proxier.getInfo.getFamilyInfo(url)
	if len(result) == 0 {
		return nil, fmt.Errorf("Cannot get FamilyInfo")
	}

	var families []familyInfo
	jsonText, err := json.Marshal(result)
	if err != nil {
		klog.V(5).Infof("Marshal cannot encoding from %v because %v", nodeAddress, err)
		return nil, err
	}
	if err := json.Unmarshal(jsonText, &families); err != nil {
		klog.V(5).Infof("JSON-encoded data cannot be parsed because %v", err)
		return nil, err
	}

	for _, f := range families {
		if f.Name == ipmiTemperatureKey {
			ambient, CPU1, CPU2 := float64(-1), float64(-1), float64(-1)
			var ambientErr, CPU1Err, CPU2Err error
			for _, m := range f.Metrics {
				if m.Labels.Name == ambientKey {
					ambient, ambientErr = strconv.ParseFloat(m.Value, 32)
				}
				if m.Labels.Name == cpu1Key {
					CPU1, CPU1Err = strconv.ParseFloat(m.Value, 32)
				}
				if m.Labels.Name == cpu2Key {
					CPU2, CPU2Err = strconv.ParseFloat(m.Value, 32)
				}

				if ambient != -1 && CPU1 != -1 && CPU2 != -1 {
					break
				}
			}

			if ambientErr != nil || CPU1Err != nil || CPU2Err != nil {
				klog.V(5).Infof("Convert to float64 failed from %v [Ambient : %v, CPU1 : %v, CPU2 : %v]", nodeAddress, ambientErr, CPU1Err, CPU2Err)
				return nil, fmt.Errorf("Convert to float64 failed")
			}
			if ambient == -1 || CPU1 == -1 || CPU2 == -1 {
				klog.V(5).Infof("Cannot get node temperature from %v [Ambient : %v, CPU1 : %v, CPU2 : %v]", nodeAddress, ambient, CPU1, CPU2)
				return nil, fmt.Errorf("Cannot get node temperature")
			}
			return []float32{float32(ambient), float32(CPU1), float32(CPU2)}, nil
		}
	}

	klog.V(5).Infof("%v is not exits from %v", ipmiTemperatureKey, nodeAddress)
	return nil, fmt.Errorf("%v is not exits from %v", ipmiTemperatureKey, nodeAddress)
}

// Get list of nodes Name inside Cluster
func (proxier *Proxier) getNodesName() {
	nodesName := []string{}
	nodes, err := proxier.clientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Warningf("Cannot get list of nodes. Error : %v", err)
	}

	for _, node := range nodes.Items {
		for _, nodeStatus := range node.Status.Conditions {
			if nodeStatus.Type == v1.NodeReady && nodeStatus.Status == v1.ConditionTrue {
				nodesName = append(nodesName, node.Name)
			}
		}
	}
	proxier.nodesName = nodesName
}

// Get list of pods endpoint inside Cluster
func (proxier *Proxier) getPodsEndpoint() {
	endpointsBelongNode := make(map[string]string)

	pods, err := proxier.clientSet.CoreV1().Pods("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Warningf("Cannot get list of pods. Error : %v", err)
	}

	for _, pod := range pods.Items {
		if pod.Status.Phase == v1.PodRunning {
			endpointsBelongNode[pod.Status.PodIP] = pod.Spec.NodeName
		}
	}
	proxier.endpointsBelongNode = endpointsBelongNode
}

// Calclate node CPU usage
func (proxier *Proxier) calcCPUUsage(nodeInfo *v1.Node) (float32, error) {
	klog.V(5).Infof("%v : Start calcCPUUsage() function", nodeInfo.Name)

	nodeMetrics, err := proxier.getInfo.getNodeMetrics(context.TODO(), nodeInfo.Name)
	if err != nil {
		klog.Errorf("%v : Cannot get metrics infomations because %v", nodeInfo.Name, err)
		return -1, err
	}

	nodeMetricsCPU := nodeMetrics.Usage["cpu"]
	nodeCPUUsage, _ := strconv.ParseFloat(nodeMetricsCPU.AsDec().String(), 32)
	nodeResource := nodeInfo.Status.Capacity["cpu"]
	nodeCPUCapacity, _ := strconv.ParseFloat(nodeResource.AsDec().String(), 32)

	CPUUsage := float32(nodeCPUUsage / nodeCPUCapacity)
	klog.V(5).Infof("%v : CPUusage %v", nodeInfo.Name, CPUUsage)

	return CPUUsage, nil
}

// Calculate normalized node temperature.
func (proxier *Proxier) calcNormalizeTemperature(nodeInfo *v1.Node) (float32, float32, float32, error) {
	klog.V(5).Infof("%v : Start calcNormalizeTemperature() function", nodeInfo.Name)

	ambientMax, ambientMin, err := getStandardTemperature(nodeInfo, labelAmbientMax, labelAmbientMin)
	if err != nil {
		return -1, -1, -1, err
	}

	CPU1Max, CPU1Min, err := getStandardTemperature(nodeInfo, labelCPU1Max, labelCPU1Min)
	if err != nil {
		return -1, -1, -1, err
	}

	CPU2Max, CPU2Min, err := getStandardTemperature(nodeInfo, labelCPU2Max, labelCPU2Min)
	if err != nil {
		return -1, -1, -1, err
	}

	if proxier.temperatureInfoBelongNode[nodeInfo.Name].ambientTimestamp.IsZero() ||
		int(time.Since(proxier.temperatureInfoBelongNode[nodeInfo.Name].ambientTimestamp).Minutes()) >= getTemperatureInterval {
		nodeIPAddress, err := getNodeInternalIP(nodeInfo)
		if err == nil {
			temp, err := proxier.getNodeTemperature(nodeIPAddress)
			if err == nil {
				nodeTemperatureInfo := nodeTemperatureInfo{
					ambient:          temp[0],
					cpu1:             temp[1],
					cpu2:             temp[2],
					ambientTimestamp: time.Now(),
				}
				proxier.temperatureInfoBelongNode[nodeInfo.Name] = nodeTemperatureInfo
			}
		} else {
			klog.Warningf("%v : Cannot get InternalIP.", nodeInfo.Name)
		}
	}

	ambient := proxier.temperatureInfoBelongNode[nodeInfo.Name].ambient
	cpu1 := proxier.temperatureInfoBelongNode[nodeInfo.Name].cpu1
	cpu2 := proxier.temperatureInfoBelongNode[nodeInfo.Name].cpu2

	if ambient == 0 || cpu1 == 0 || cpu2 == 0 {
		klog.Warningf("%v : not exist ipmi data", nodeInfo.Name)
		return -1, -1, -1, errors.New("not exist ipmi data")
	}

	klog.V(5).Infof("%v : temperature [ambient: %v ℃, CPU1: %v ℃, CPU2: %v ℃]", nodeInfo.Name, ambient, cpu1, cpu2)
	normalizedAmbient := (ambient - ambientMin) / (ambientMax - ambientMin)
	normalizedCPU1 := (cpu1 - CPU1Min) / (CPU1Max - CPU1Min)
	normalizedCPU2 := (cpu2 - CPU2Min) / (CPU2Max - CPU2Min)
	return normalizedAmbient, normalizedCPU1, normalizedCPU2, nil
}

// Score calculates node score.
// The returned score is the amount of increase in current power consumption.
func (proxier *Proxier) Score(nodeName string) int64 {
	klog.V(5).Infof("%v : Start Score() function", nodeName)

	nodeInfo, err := proxier.clientSet.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		klog.Errorf("%v : Cannot get Nodes info. Error : %v", nodeName, err)
		return -1
	}

	CPUUsage, err := proxier.calcCPUUsage(nodeInfo)
	if err != nil {
		return -1
	}

	var preInput predictInput
	ambientTemp, cpu1Temp, cpu2Temp, err := proxier.calcNormalizeTemperature(nodeInfo)
	if err != nil {
		return -1
	}

	preInput.ambientTemp = float32(math.Round(float64(ambientTemp)*10) / 10)
	preInput.cpu1Temp = float32(math.Round(float64(cpu1Temp)*10) / 10)
	preInput.cpu2Temp = float32(math.Round(float64(cpu2Temp)*10) / 10)
	preInput.cpuUsage = float32(math.Round(float64(CPUUsage)*10) / 10)
	klog.V(5).Infof("%v : predict params %+v", nodeName, preInput)

	return proxier.calcScore(nodeInfo, preInput)
}

// Calculate the score from CPU usage.
func (proxier *Proxier) calcScore(nodeInfo *v1.Node, preInput predictInput) int64 {
	klog.V(5).Infof("%v : Start calcScore() function", nodeInfo.Name)
	var score float32
	var err error
	if v, ok := proxier.powerConsumptionCache.getPCCache(preInput, nodeInfo.Name); ok && v != -1 {
		klog.V(5).Infof("%v : Use cache %+v=%v", nodeInfo.Name, preInput, v)
		score = v
	} else {
		score, err = proxier.getInfo.predictPC(preInput, nodeInfo)
		if err != nil {
			klog.Warningf("%v : Cannnot predict power consumption because %v", nodeInfo.Name, err)
			return -1
		}
		klog.V(5).Infof("%v : predictPC result %+v=%v", nodeInfo.Name, preInput, score)
		proxier.powerConsumptionCache.setPCCache(preInput, nodeInfo.Name, score)
	}

	return int64(score)
}
