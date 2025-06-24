### 📘 **README.md**

```markdown
# Azure Pupil Detection Function

This project implements a serverless Azure Function to perform real-time pupil detection from image frames using OpenCV and ONNX-based deep learning models. It is designed for fast deployment in Azure environments and can be used in eye-tracking applications, accessibility tools, or attention monitoring systems.

---

## 🚀 Features

- Azure Function-based architecture
- Frame-by-frame pupil detection
- Uses pre-trained ONNX model for deep learning inference
- Supports Haar Cascade for eye detection
- JSON input/output handling
- Designed for scalable, event-driven workloads in Azure

---

## 🛠 Project Structure

```

.
├── Program.cs                  # Azure Functions bootstrap
├── ProcessPupilFrame.cs       # Core logic for frame processing
├── PupilDetectionFunction.csproj # Project configuration
├── Models/
│   ├── haarcascade\_eye.xml    # Haar cascade XML for eye detection
│   └── model.onnx             # ONNX deep learning model
├── host.json                  # Azure Functions host configuration
├── local.settings.json        # Local dev settings (excluded from deployment)

````

---

## 📦 Requirements

- [.NET 6 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/6.0)
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local)

---

## 🔧 Setup & Deployment

### Run Locally

```bash
func start
````

### Deploy to Azure

```bash
func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
```

---

## 📥 Input

The function expects a JSON payload containing a base64-encoded image frame:

```json
{
  "image": "<base64-encoded-frame>"
}
```

---

## 📤 Output

The function returns the coordinates of detected pupils:

```json
{
  "pupils": [
    { "x": 123, "y": 145, "width": 30, "height": 30 },
    ...
  ]
}
```

---

## 🧠 Model Info

* `model.onnx`: Deep learning model for fine pupil localization.
* `haarcascade_eye.xml`: Used for rapid detection of eye regions before detailed inference.

---

