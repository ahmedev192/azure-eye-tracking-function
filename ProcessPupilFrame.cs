using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PupilDetectionFunction
{
    public class ProcessPupilFrame
    {
        private readonly ILogger<ProcessPupilFrame> _logger;
        private readonly CascadeClassifier _eyeCascade;
        private readonly InferenceSession _onnxSession;
        private readonly List<double> _pupilDiameters = new();
        private readonly List<double> _savedPupilDiameters = new();

        public ProcessPupilFrame(ILoggerFactory loggerFactory)
        {
            _logger = loggerFactory.CreateLogger<ProcessPupilFrame>();

            // Initialize Haar cascade
            string cascadePath = Path.Combine(Directory.GetCurrentDirectory(), "Models", "haarcascade_eye.xml");
            _eyeCascade = new CascadeClassifier(cascadePath);
            if (_eyeCascade.Empty())
            {
                _logger.LogError("Failed to load Haar cascade file.");
                throw new Exception("Failed to load Haar cascade file.");
            }

            // Initialize ONNX model
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "Models", "model.onnx");
            _onnxSession = new InferenceSession(modelPath);
        }

        [Function("ProcessPupilFrame")]
        public async Task<HttpResponseData> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "post")] HttpRequestData req)
        {
            _logger.LogInformation("Processing pupil detection frame.");

            // Read image from request (base64 encoded)
            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            var requestData = JsonSerializer.Deserialize<RequestData>(requestBody);
            if (string.IsNullOrEmpty(requestData.ImageBase64) && !requestData.SaveDiameters)
            {
                var errorResponse = req.CreateResponse(System.Net.HttpStatusCode.BadRequest);
                await errorResponse.WriteStringAsync("Missing or invalid 'imageBase64' in request body.");
                return errorResponse;
            }

            // Create RIPARealTime instance for this request
            var ripaRealTime = new RIPARealTime(6, 6, 3, 100, 1.0);

            // Process frame, passing requestData
            var result = ProcessFrame(requestData, ripaRealTime);

            // Create response
            var response = req.CreateResponse(System.Net.HttpStatusCode.OK);
            response.Headers.Add("Content-Type", "application/json");
            await response.WriteStringAsync(JsonSerializer.Serialize(result));
            return response;
        }

        private object ProcessFrame(RequestData requestData, RIPARealTime ripaRealTime)
        {
            string diametersOutput = "";
            double totalPupilDiameter = 0;
            int pupilCount = 0;
            string processedImageBase64 = "";
            Mat matFrame = null;

            // Process image if provided
            if (!string.IsNullOrEmpty(requestData.ImageBase64))
            {
                byte[] imageBytes;
                try
                {
                    imageBytes = Convert.FromBase64String(requestData.ImageBase64);
                }
                catch (Exception ex)
                {
                    _logger.LogError($"Invalid base64 image: {ex.Message}");
                    return new { Error = $"Invalid base64 image: {ex.Message}" };
                }

                // Convert to Mat
                matFrame = Cv2.ImDecode(imageBytes, ImreadModes.Color);
                if (matFrame.Empty())
                {
                    _logger.LogError("Failed to decode image.");
                    return new { Error = "Failed to decode image." };
                }

                // Detect eyes
                OpenCvSharp.Rect[] eyes = _eyeCascade.DetectMultiScale(matFrame, scaleFactor: 1.3, minNeighbors: 5, minSize: new OpenCvSharp.Size(30, 30));

                int maxEyesToProcess = 2;
                int eyesProcessed = 0;

                foreach (var eye in eyes)
                {
                    if (eyesProcessed >= maxEyesToProcess)
                        break;

                    // Draw rectangle around eye
                    Cv2.Rectangle(matFrame, eye, new Scalar(255, 0, 0), 2);

                    // Extract and preprocess eye region
                    using Mat eyeRegion = new Mat(matFrame, eye);
                    using Mat resizedEyeRegion = new Mat();
                    Cv2.Resize(eyeRegion, resizedEyeRegion, new OpenCvSharp.Size(128, 128));

                    using Mat grayEyeRegion = new Mat();
                    Cv2.CvtColor(resizedEyeRegion, grayEyeRegion, ColorConversionCodes.BGR2GRAY);
                    grayEyeRegion.ConvertTo(grayEyeRegion, MatType.CV_32F, 1.0 / 255.0);

                    float[] inputData = new float[128 * 128];
                    grayEyeRegion.GetArray(out inputData);

                    // Prepare ONNX input tensor
                    var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 128, 128, 1 });
                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_1", inputTensor) };

                    // Run inference
                    using var results = _onnxSession.Run(inputs);
                    var outputTensor = results.First().AsTensor<float>();
                    var heatmap = outputTensor.ToArray();
                    int heatmapWidth = outputTensor.Dimensions[3];
                    int heatmapHeight = outputTensor.Dimensions[2];

                    // Find max value in heatmap
                    double maxVal = double.MinValue;
                    int maxX = 0, maxY = 0;
                    for (int y = 0; y < heatmapHeight; y++)
                    {
                        for (int x = 0; x < heatmapWidth; x++)
                        {
                            double value = heatmap[y * heatmapWidth + x];
                            if (value > maxVal)
                            {
                                maxVal = value;
                                maxX = x;
                                maxY = y;
                            }
                        }
                    }

                    // Calculate pupil center
                    double pupilX = maxX * eye.Width / (double)heatmapWidth + eye.X;
                    double pupilY = maxY * eye.Height / (double)heatmapHeight + eye.Y;
                    Cv2.Circle(matFrame, new Point(pupilX, pupilY), 5, new Scalar(0, 255, 0), -1);

                    // Assume second output is pupil area (adjust based on your model)
                    float pupilArea = results.Count > 1 ? results[1].AsTensor<float>()[0] : 0;
                    if (pupilArea > 0)
                    {
                        double pupilDiameter = 2 * Math.Sqrt(pupilArea / Math.PI);
                        _pupilDiameters.Add(pupilDiameter);
                        totalPupilDiameter += pupilDiameter;
                        pupilCount++;
                        diametersOutput += $"{pupilDiameter:F2} ";
                    }
                    else
                    {
                        _logger.LogWarning("Unexpected model output length.");
                    }

                    eyesProcessed++;
                }

                // Encode processed image
                if (matFrame != null)
                {
                    byte[] processedImageBytes = matFrame.ImEncode(".jpg");
                    processedImageBase64 = Convert.ToBase64String(processedImageBytes);
                }
            }

            // Calculate average and process with RIPA
            double? averagePupilDiameter = null;
            double? ripaValue = null;
            if (pupilCount > 0)
            {
                averagePupilDiameter = totalPupilDiameter / pupilCount;
            }

            // Process the provided AveragePupilDiameterBuffer
            if (requestData.AveragePupilDiameterBuffer != null && requestData.AveragePupilDiameterBuffer.Any())
            {
                foreach (var diameter in requestData.AveragePupilDiameterBuffer)
                {
                    ripaValue = ripaRealTime.ProcessData(diameter);
                }
                // Use the last ripaValue (most recent)
                if (averagePupilDiameter.HasValue)
                {
                    ripaValue = ripaRealTime.ProcessData(averagePupilDiameter.Value);
                }
            }
            else if (averagePupilDiameter.HasValue)
            {
                ripaValue = ripaRealTime.ProcessData(averagePupilDiameter.Value);
            }

            // Prepare response data
            var result = new
            {
                PupilDiameters = diametersOutput.Trim(),
                TotalPupilDiameter = totalPupilDiameter.ToString("F2"),
                AveragePupilDiameter = averagePupilDiameter?.ToString("F2"),
                RipaValue = ripaValue.HasValue && !double.IsNaN(ripaValue.Value) ? ripaValue.Value.ToString("F2") : null,
                ProcessedImageBase64 = processedImageBase64,
                SavedPupilDiameters = requestData.SaveDiameters ? SavePupilDiameters() : null
            };

            return result;
        }

        private double[] SavePupilDiameters()
        {
            _savedPupilDiameters.AddRange(_pupilDiameters);
            _logger.LogInformation("Pupil diameters saved.");
            var saved = _savedPupilDiameters.ToArray();
            _pupilDiameters.Clear();
            return saved;
        }

        // Model for deserializing request body
        private class RequestData
        {
            public string ImageBase64 { get; set; } = string.Empty;
            public bool SaveDiameters { get; set; }
            public double[] AveragePupilDiameterBuffer { get; set; } = Array.Empty<double>();
        }
    }

    // RIPARealTime and SavitzkyGolayFilter classes (unchanged)
    public class RIPARealTime
    {
        private readonly int bufferSize;
        private readonly double thresholdFactor;
        private readonly Queue<double> rawBuffer;
        private readonly Queue<double> lowPassBuffer;
        private readonly Queue<double> highPassBuffer;
        private readonly Queue<double> ratioBuffer;
        private readonly SavitzkyGolayFilter smoothingFilter;
        private readonly SavitzkyGolayFilter lowPassFilter;
        private readonly SavitzkyGolayFilter highPassFilter;

        public RIPARealTime(int nLeft, int nRight, int polyOrder, int bufferSize, double thresholdFactor)
        {
            this.bufferSize = bufferSize;
            this.thresholdFactor = thresholdFactor;
            this.rawBuffer = new Queue<double>(bufferSize);
            this.lowPassBuffer = new Queue<double>(bufferSize);
            this.highPassBuffer = new Queue<double>(bufferSize);
            this.ratioBuffer = new Queue<double>(bufferSize);
            this.smoothingFilter = new SavitzkyGolayFilter(nLeft, nRight, polyOrder);
            this.lowPassFilter = new SavitzkyGolayFilter(nLeft + 2, nRight + 2, 2);
            this.highPassFilter = new SavitzkyGolayFilter(nLeft - 1, nRight - 1, 1);
        }

        public double ProcessData(double newValue)
        {
            double smoothedValue = smoothingFilter.FilterNextValue(newValue);
            if (rawBuffer.Count == bufferSize)
                rawBuffer.Dequeue();
            rawBuffer.Enqueue(smoothedValue);

            double lowPassValue = 0;
            double highPassValue = 0;

            System.Threading.Tasks.Parallel.Invoke(
                () => { lowPassValue = lowPassFilter.FilterNextValue(smoothedValue); },
                () => { highPassValue = highPassFilter.FilterNextValue(smoothedValue); }
            );

            if (lowPassBuffer.Count == bufferSize)
                lowPassBuffer.Dequeue();
            lowPassBuffer.Enqueue(lowPassValue);

            if (highPassBuffer.Count == bufferSize)
                highPassBuffer.Dequeue();
            highPassBuffer.Enqueue(highPassValue);

            double ratio = lowPassValue / (Math.Abs(highPassValue) + 1e-6);
            if (ratioBuffer.Count == bufferSize)
                ratioBuffer.Dequeue();
            ratioBuffer.Enqueue(ratio);

            if (rawBuffer.Count < bufferSize)
                return double.NaN;

            double normalizedRatio = NormalizeRatio(ratio);
            double normalizedRIPA = NormalizeRIPA(ComputeRIPA());
            return normalizedRIPA;
        }

        private double NormalizeRatio(double ratio)
        {
            if (ratioBuffer.Count == 0) return ratio;
            double minRatio = double.MaxValue;
            double maxRatio = double.MinValue;
            foreach (double r in ratioBuffer)
            {
                if (r < minRatio) minRatio = r;
                if (r > maxRatio) maxRatio = r;
            }
            return (ratio - minRatio) / (maxRatio - minRatio + 1e-6);
        }

        private double ComputeRIPA()
        {
            double[] ratioArray = ratioBuffer.ToArray();
            double median = Median(ratioArray);
            double stdDev = StandardDeviation(ratioArray);
            double threshold = median + thresholdFactor * stdDev;

            int peakCount = 0;
            for (int i = 1; i < ratioArray.Length - 1; i++)
            {
                if (ratioArray[i] > threshold || ratioArray[i] > ratioArray[i - 1] || ratioArray[i] > ratioArray[i + 1])
                {
                    peakCount++;
                }
            }
            return peakCount;
        }

        private double NormalizeRIPA(double ripaValue)
        {
            return 1.0 - (ripaValue / bufferSize);
        }

        private double Median(double[] values)
        {
            Array.Sort(values);
            int mid = values.Length / 2;
            return values.Length % 2 == 0 ? (values[mid - 1] + values[mid]) / 2.0 : values[mid];
        }

        private double StandardDeviation(double[] values)
        {
            double mean = values.Average();
            double variance = values.Sum(value => Math.Pow(value - mean, 2)) / values.Length;
            return Math.Sqrt(variance);
        }
    }

    public class SavitzkyGolayFilter
    {
        private readonly int nLeft;
        private readonly int nRight;
        private readonly int polyOrder;
        private readonly double[] coefficients;
        private readonly Queue<double> buffer;

        public SavitzkyGolayFilter(int nLeft, int nRight, int polyOrder)
        {
            this.nLeft = nLeft;
            this.nRight = nRight;
            this.polyOrder = polyOrder;
            this.buffer = new Queue<double>(nLeft + nRight + 1);
            this.coefficients = CalculateCoefficients(nLeft, nRight, polyOrder);
        }

        public double FilterNextValue(double newValue)
        {
            if (buffer.Count == nLeft + nRight + 1)
                buffer.Dequeue();
            buffer.Enqueue(newValue);

            if (buffer.Count < nLeft + nRight + 1)
                return newValue;

            double[] data = buffer.ToArray();
            double result = 0;
            for (int i = 0; i < data.Length; i++)
            {
                result += data[i] * coefficients[i];
            }
            return result;
        }

        private double[] CalculateCoefficients(int nLeft, int nRight, int polyOrder)
        {
            int windowSize = nLeft + nRight + 1;
            double[,] A = new double[windowSize, windowSize];
            double[] b = new double[windowSize];
            b[nLeft] = 1.0;

            for (int i = 0; i < windowSize; i++)
            {
                for (int j = 0; j < windowSize; j++)
                {
                    A[i, j] = Math.Pow(i - nLeft, j);
                }
            }

            return GaussianElimination(A, b);
        }

        private double[] GaussianElimination(double[,] A, double[] b)
        {
            int n = b.Length;
            double[] x = new double[n];

            for (int i = 0; i < n; i++)
            {
                int max = i;
                for (int k = i + 1; k < n; k++)
                {
                    if (Math.Abs(A[k, i]) > Math.Abs(A[max, i]))
                        max = k;
                }

                for (int k = i; k < n; k++)
                {
                    double tmp = A[max, k];
                    A[max, k] = A[i, k];
                    A[i, k] = tmp;
                }
                double tmpB = b[max];
                b[max] = b[i];
                b[i] = tmpB;

                for (int k = i + 1; k < n; k++)
                {
                    double factor = A[k, i] / A[i, i];
                    b[k] -= factor * b[i];
                    for (int j = i; j < n; j++)
                    {
                        A[k, j] -= factor * A[i, j];
                    }
                }
            }

            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0.0;
                for (int j = i + 1; j < n; j++)
                {
                    sum += A[i, j] * x[j];
                }
                x[i] = (b[i] - sum) / A[i, i];
            }

            return x;
        }
    }
}