# 📊 Monitor.py - Complete Guide

Sistem monitoring fleksibel untuk logging variabel, tensor, dan proses dengan output ke console dan file.

---

## 🚀 Quick Start

```python
from monitor import Monitor

# Basic usage
with Monitor(session_name="my_experiment") as monitor:
    monitor.log("Starting experiment...")
    monitor.log_variable("learning_rate", 0.001)
    
    with monitor.step("Training"):
        # Your code here
        monitor.log_tensor("weights", tensor)
```

**Output:**
- Console: Progress messages
- File: `./logs/monitor_my_experiment.log` (detailed logs)

---

## 📁 File Structure

```
project/
├── monitor.py              # Main monitoring system
├── example_llm_monitor.py  # LLM-specific examples
├── logs/                   # Auto-created log directory
│   ├── monitor_*.log       # Log files dengan timestamp
│   └── ...
```

---

## 🔧 Configuration

```python
from monitor import Monitor, MonitorConfig

config = MonitorConfig(
    log_dir="./logs",              # Directory untuk log files
    log_file_prefix="monitor",     # Prefix nama file
    console_output=True,           # Print ke console?
    file_output=True,              # Write ke file?
    max_tensor_elements=100,       # Max elemen tensor ditampilkan
    tensor_precision=4,            # Decimal precision
    auto_create_dir=True,          # Auto-create log directory?
    timestamp_format="%Y%m%d_%H%M%S",  # Format timestamp
)

monitor = Monitor(config, session_name="custom_session")
```

---

## 📝 Core Methods

### 1. **Simple Logging**

```python
monitor.log("This is a message")
monitor.log("Console only", to_file=False)
monitor.log("File only", to_console=False)
```

### 2. **Log Variables (Auto-Format)**

```python
# Primitives
monitor.log_variable("learning_rate", 0.001)
monitor.log_variable("epoch", 10)
monitor.log_variable("model_name", "GPT-4")

# Tensor (auto-detected)
monitor.log_variable("weights", torch.randn(3, 4))

# Dictionary
config = {"lr": 0.001, "batch_size": 32}
monitor.log_variable("config", config)

# List
monitor.log_variable("losses", [0.5, 0.3, 0.2, 0.1])
```

### 3. **Log Tensor (Specialized)**

```python
tensor = torch.randn(100, 512, 4096)

# Default
monitor.log_tensor("hidden_state", tensor)

# Custom max elements
monitor.log_tensor("weights", tensor, max_elements=50)

# File only (no console spam)
monitor.log_tensor("large_tensor", tensor, to_console=False)
```

**Output Format:**
```
hidden_state =
  Tensor(shape=[100, 512, 4096], dtype=torch.float32, device=cpu)
  Stats: min=-3.8234, max=3.9123, mean=0.0012, std=1.0001
  Values: [-0.5234, 1.2341, ..., 0.8765]
```

### 4. **Log Dictionary**

```python
results = {
    "accuracy": 0.95,
    "loss": 0.123,
    "metrics": {"f1": 0.92, "precision": 0.94}
}

monitor.log_dict("evaluation_results", results)
```

**Output:**
```json
evaluation_results =
  {
    "accuracy": 0.95,
    "loss": 0.123,
    "metrics": {
      "f1": 0.92,
      "precision": 0.94
    }
  }
```

### 5. **Log Separator**

```python
monitor.log_separator()                    # ──────────────────────
monitor.log_separator(char="=", length=80) # ════════════════════
```

---

## 🔄 Step Tracking (Context Manager)

```python
with monitor.step("Data Loading"):
    monitor.log("Loading dataset...")
    # Your code here
    monitor.log_variable("num_samples", 10000)

# Nested steps
with monitor.step("Training"):
    for epoch in range(3):
        with monitor.step(f"Epoch {epoch}"):
            # Training code
            monitor.log_variable("loss", loss)
```

**Output:**
```
[  0.123s] ────────────────────────────────────────────────────
[  0.124s] ▶ START: Data Loading
[  0.125s]   Loading dataset...
[  0.156s]   num_samples = 10000
[  0.157s] ◀ END: Data Loading (took 0.033s)
```

---

## 🎯 Function Tracking (Decorator)

```python
@monitor.track_function
def train_model(data, labels):
    """Your training logic"""
    time.sleep(0.1)
    return {"loss": 0.5, "accuracy": 0.9}

# Auto-logged when called
result = train_model(X, y)
```

**Log Output:**
```
▶ CALL: train_model()
  args = (tensor(...), tensor(...))
  kwargs = {}
  return = {'loss': 0.5, 'accuracy': 0.9}
◀ RETURN: train_model() (took 0.105s)
```

---

## 💡 Common Use Cases

### Use Case 1: Monitor LLM Generation (Console + File)

```python
config = MonitorConfig(
    console_output=True,   # Progress di console
    file_output=True,      # Detail di file
)

with Monitor(config, session_name="generation") as monitor:
    
    # Console: Progress
    monitor.log("Generating text...", to_console=True)
    
    # File only: Detailed tensors
    for step in range(max_tokens):
        monitor.log_tensor(
            f"step_{step}_logits", 
            logits, 
            to_console=False,  # ← No console spam
            to_file=True,      # ← Saved to file
        )
        
        # Console: Token generated
        monitor.log(f"Token: {token_text}", to_console=True)
```

### Use Case 2: Silent Logging (File Only)

```python
# Untuk proses dengan banyak iterasi
config = MonitorConfig(console_output=False, file_output=True)

with Monitor(config, session_name="training") as monitor:
    for epoch in range(1000):
        # Tidak muncul di console
        monitor.log_variable(f"epoch_{epoch}_loss", loss)
        monitor.log_tensor(f"epoch_{epoch}_weights", weights)
    
    print("✓ All logs saved to file")
```

### Use Case 3: Debugging Tensor Shapes

```python
with monitor.step("Forward Pass"):
    x = torch.randn(32, 512)
    monitor.log_variable("input_shape", x.shape)
    
    x = model.layer1(x)
    monitor.log_variable("after_layer1", x.shape)
    
    x = model.layer2(x)
    monitor.log_variable("after_layer2", x.shape)
    
    # Quick check tanpa print()
    # Semua tersimpan di log file
```

### Use Case 4: Compare Experiments

```python
# Experiment 1
with Monitor(session_name="exp_lr_0.001") as m:
    # ... training with lr=0.001
    m.log_dict("final_metrics", metrics)

# Experiment 2
with Monitor(session_name="exp_lr_0.01") as m:
    # ... training with lr=0.01
    m.log_dict("final_metrics", metrics)

# Compare:
# logs/monitor_exp_lr_0.001.log
# logs/monitor_exp_lr_0.01.log
```

---

## 🎨 Formatting Examples

### Auto-Format Tensor

**Input:**
```python
tensor = torch.randn(2, 3)
monitor.log_tensor("weights", tensor)
```

**Output:**
```
weights =
  Tensor(shape=[2, 3], dtype=torch.float32, device=cpu)
  Stats: min=-1.5234, max=2.1234, mean=0.0123, std=1.0234
  Values: [-1.5234, 0.8765, 2.1234, -0.3456, 0.9876, 1.2345]
```

### Auto-Format Large Tensor

**Input:**
```python
tensor = torch.randn(1000)
monitor.log_tensor("large", tensor, max_elements=10)
```

**Output:**
```
large =
  Tensor(shape=[1000], dtype=torch.float32, device=cpu)
  Stats: min=-3.2, max=3.5, mean=0.01, std=1.0
  Values: [-0.12, 1.34, 0.56, -2.11, 0.89 ... (990 more) ... 1.23, -0.45, 2.67, 0.12, -1.34]
```

### Auto-Format Dictionary

**Input:**
```python
config = {
    "model": "GPT-4",
    "params": {"lr": 0.001, "batch": 32},
    "enabled": True
}
monitor.log_dict("config", config)
```

**Output:**
```
config =
  {
    "model": "GPT-4",
    "params": {
      "lr": 0.001,
      "batch": 32
    },
    "enabled": true
  }
```

---

## 🌐 Global Monitor (Optional)

Untuk convenience, bisa pakai global monitor:

```python
from monitor import log, log_variable, log_tensor, step

# Auto-create global instance
log("Starting...")
log_variable("x", 10)
log_tensor("weights", tensor)

with step("Training"):
    log("Training model...")
```

---

## 📂 Log File Format

```
================================================================================
Monitor Log Session Started
Time: 2025-03-18 14:23:45
================================================================================
[  0.000s] Starting experiment...
[  0.001s] learning_rate = 0.001
[  0.002s] ────────────────────────────────────────────────────────────────
[  0.003s] ▶ START: Data Loading
[  0.004s]   Loading dataset...
[  0.156s]   num_samples = 10000
[  0.157s] ◀ END: Data Loading (took 0.153s)
[  0.158s] ────────────────────────────────────────────────────────────────
...
================================================================================
Monitor Log Session Ended
Total Duration: 12.45s
================================================================================
```

---

## 🔍 Tips & Best Practices

### 1. **Console vs File**

```python
# Progress di console, detail di file
monitor.log("Training epoch 1/10", to_console=True)
monitor.log_tensor("gradients", grads, to_console=False, to_file=True)
```

### 2. **Nested Steps untuk Struktur**

```python
with monitor.step("Experiment"):
    with monitor.step("Training"):
        with monitor.step("Epoch 1"):
            # Automatically indented in log
            monitor.log("Batch 1/100")
```

### 3. **Conditional Logging**

```python
# Log detail hanya untuk debug
DEBUG = True

if DEBUG:
    monitor.log_tensor("intermediate_activations", activations, to_console=False)
else:
    monitor.log("Step completed")
```

### 4. **Session Names untuk Organization**

```python
# Mudah identify eksperimen
Monitor(session_name="baseline_v1")
Monitor(session_name="with_dropout_0.5")
Monitor(session_name="doubled_lr")
```

### 5. **Function Tracking untuk Profiling**

```python
@monitor.track_function
def expensive_operation(data):
    # Auto-log execution time
    time.sleep(2)
    return result

# Log akan show: "took 2.003s"
```

---

## ⚙️ Advanced Usage

### Custom Formatter

```python
from monitor import Formatter

# Extend formatter
class CustomFormatter(Formatter):
    @staticmethod
    def format_my_type(obj):
        return f"Custom: {obj}"
```

### Multiple Monitors

```python
# Different configs untuk different purposes
train_monitor = Monitor(
    config=MonitorConfig(console_output=False),
    session_name="training"
)

debug_monitor = Monitor(
    config=MonitorConfig(max_tensor_elements=1000),
    session_name="debug"
)
```

### Context Manager Chain

```python
with Monitor(session_name="exp1") as m1:
    with m1.step("Phase 1"):
        # ...
        pass
    
    with m1.step("Phase 2"):
        # ...
        pass
```

---

## 🐛 Troubleshooting

**Q: Log file tidak dibuat?**
```python
# Check config
config = MonitorConfig(
    file_output=True,        # ← Harus True
    auto_create_dir=True,    # ← Auto-create directory
)
```

**Q: Tensor terlalu besar di log?**
```python
# Kurangi max_elements
monitor.log_tensor("huge_tensor", tensor, max_elements=20)
```

**Q: Console spam dengan tensor?**
```python
# File only
monitor.log_tensor("tensor", tensor, to_console=False)
```

---

## 📚 Full Example: LLM Generation

```python
from monitor import Monitor, MonitorConfig

config = MonitorConfig(
    console_output=True,
    file_output=True,
    max_tensor_elements=50,
)

with Monitor(config, session_name="llm_gen") as monitor:
    
    # Setup
    with monitor.step("Model Loading"):
        model = load_model()
        monitor.log("✓ Model loaded")
    
    # Generation loop
    with monitor.step("Generation"):
        for step in range(10):
            
            # Progress di console
            monitor.log(f"Step {step+1}/10", to_console=True)
            
            # Detail di file
            with monitor.step(f"Step_{step}", to_console=False):
                outputs = model(input_ids)
                
                # Log tensors ke file saja
                monitor.log_tensor("logits", outputs.logits, to_console=False)
                monitor.log_variable("logits_shape", outputs.logits.shape, to_console=False)
                
                # Log token yang dipilih ke console
                token = select_token(outputs.logits)
                monitor.log(f"Generated: {token}", to_console=True)
    
    # Summary
    monitor.log_separator()
    monitor.log(f"✓ Generation complete")

print(f"Detailed log: {monitor.log_file_path}")
```

---

## 📄 License

MIT License - Use freely, modify as needed.

---

## 🤝 Contributing

Suggestions welcome! File ini dirancang untuk fleksibilitas maksimal.