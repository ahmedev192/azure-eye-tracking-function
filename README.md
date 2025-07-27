# 👁️ Azure Pupil Detection Function

A serverless Azure Function that performs real-time pupil detection from image frames using OpenCV and ONNX-based deep learning models. Ideal for integration into eye-tracking systems, accessibility platforms, or attention-monitoring applications.

---

## 🚀 Key Features

- ⚡ Serverless deployment using Azure Functions
- 🧠 Pupil detection with ONNX deep learning inference
- 🔍 Eye region pre-detection using Haar Cascades
- 📦 JSON input/output support for API integration
- ☁️ Scalable, event-driven design for Azure cloud environments

---

## 🧾 Project Structure

```
.
├── Program.cs                     # Azure Functions bootstrap
├── ProcessPupilFrame.cs          # Core logic for pupil detection
├── PupilDetectionFunction.csproj # Project configuration
├── Models/
│   ├── haarcascade_eye.xml       # Haar Cascade for initial eye region detection
│   └── model.onnx                # ONNX model for pupil localization
├── host.json                     # Azure Functions host settings
├── local.settings.json           # Local development settings (excluded from repo)
```

---

## 📦 Requirements

- [.NET 6 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/6.0)
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local)
- Python (optional, for preprocessing or debugging)

---

## ⚙️ Setup & Deployment

### Run Locally

```bash
func start
```

### Deploy to Azure

```bash
func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
```

---

## 📥 Input Format

The function expects a **JSON payload** with a base64-encoded image frame:

```json
{
  "image": "<base64-encoded-frame>"
}
```

---

## 📤 Output Format

Returns detected pupil bounding boxes as a list of coordinate objects:

```json
{
  "pupils": [
    { "x": 123, "y": 145, "width": 30, "height": 30 },
    ...
  ]
}
```

---

## 🧠 Model Details

- `model.onnx`: A lightweight ONNX model trained for precise pupil localization.
- `haarcascade_eye.xml`: Used for efficient pre-detection of eye regions using OpenCV.

---

## 🔐 Security Note

Ensure all image data is sanitized and base64 strings are validated to prevent payload injection in production environments.

---

## 🙌 Acknowledgments

- [Azure Functions](https://azure.microsoft.com/services/functions/)
- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
