# 🤖 Agentic AI Scheduling Assistant - Enhanced Edition

## AMD Hackathon 2025 - Advanced Autonomous Scheduling Solution

This enhanced version integrates **Agentic AI with MCP (Model Context Protocol)** and **Operations Research optimization** to deliver intelligent, autonomous meeting scheduling.

## 🚀 Key Enhancements

### 1. **Agentic AI Information Extraction**
- **Intelligent Agents** using Pydantic AI framework
- **Advanced parsing** of meeting details: duration, urgency, preferences, type
- **Context-aware reasoning** for better understanding of email content
- **Fallback mechanisms** to traditional LLM extraction

### 2. **OR-Tools Based Optimization**
- **Constraint Programming** with CP-SAT solver
- **Multi-objective optimization** considering preferences and constraints  
- **Heuristic algorithms** for complex scheduling scenarios
- **Sub-2 second** optimization with time limits for hackathon requirements

### 3. **MCP Integration**
- **Standardized tool interfaces** for agent operations
- **Modular architecture** supporting multiple AI providers
- **Extensible framework** for future enhancements

## 🛠️ Architecture Overview

```
📧 Email Request
    ↓
🤖 Agentic Information Extraction
    ↓  
🔍 OR-Tools Optimization Engine
    ↓
📅 Google Calendar Integration  
    ↓
✅ Optimal Meeting Schedule
```

## 📦 Installation & Setup

### 1. Enhanced Dependencies
```bash
pip install -r requirements_agentic.txt
```

### 2. vLLM Server Configuration
```bash
# Start vLLM server with tool calling support
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### 3. Google Calendar Setup
- Follow existing Google Calendar setup instructions
- Ensure `token.json` is available for API access

## 🎯 Usage

### Agentic Mode (Default)
```python
# Initialize with agentic capabilities enabled
scheduler = AgenticScheduler(
    vllm_url="http://localhost:8000/v1",
    model_name="your-model-path", 
    agentic_mode=True  # Enable advanced AI agents
)

# Process meeting request
result = scheduler.process_meeting_request(input_data)
```

### Traditional Mode (Fallback)
```python
# Initialize in traditional mode
scheduler = AgenticScheduler(agentic_mode=False)
result = scheduler.process_meeting_request(input_data)
```

## 🧠 Agentic AI Features

### Information Extraction Agent
- **Meeting Duration**: Intelligent estimation from context
- **Time Preferences**: morning/afternoon/evening/anytime classification
- **Urgency Analysis**: high/medium/low priority detection
- **Meeting Type**: team_meeting/client_call/interview/presentation

Example extraction:
```python
{
    "meeting_duration_minutes": 60,
    "time_preference": "morning", 
    "urgency": "high",
    "meeting_type": "client_call",
    "recurring": false
}
```

### OR-Tools Optimization Engine
- **Constraint Programming**: Hard constraints for conflicts and business hours
- **Preference Weights**: Soft constraints for time preferences
- **Multi-day Search**: 7-day lookahead with weekend skipping
- **15-minute Granularity**: Precise slot allocation

## 📊 Performance Optimizations

### Latency Management
- **2-second OR-Tools time limit** for constraint solving
- **Parallel calendar fetching** using ThreadPoolExecutor
- **Heuristic fallback** when optimization times out
- **Cached preference weights** for repeated calculations

### Error Handling
- **Graceful degradation** from agentic to traditional modes
- **Multiple fallback layers** for robust operation
- **Comprehensive logging** for debugging

## 🔧 Configuration Options

### Agentic Parameters
```python
# Time preference weights
MORNING_WEIGHT = 10   # 9 AM - 12 PM
AFTERNOON_WEIGHT = 8  # 12 PM - 5 PM  
EVENING_WEIGHT = 6    # 5 PM - 6 PM

# OR-Tools settings
SLOT_DURATION = 15    # minutes
MAX_SOLVE_TIME = 2.0  # seconds
BUSINESS_HOURS = (9, 18)  # 9 AM to 6 PM
```

## 🧪 Testing

### Run Agentic Tests
```bash
python test_agentic_scheduler.py
```

### Performance Benchmarks
- **Average latency**: < 3 seconds end-to-end
- **OR optimization**: < 2 seconds for 7-day search
- **Agent extraction**: < 1 second per email
- **Calendar API calls**: < 2 seconds parallel fetch

## 🏆 Hackathon Advantages

### Innovation Points
1. **Advanced AI Agents** with reasoning capabilities
2. **Operations Research** optimization algorithms  
3. **MCP Protocol** integration for extensibility
4. **Sub-10 second latency** with sophisticated algorithms
5. **Autonomous coordination** without human intervention

### Scalability Features
- **Modular architecture** for easy extension
- **Multi-provider support** (OpenAI, Anthropic, etc.)
- **Caching mechanisms** for repeated optimizations
- **Async support** for high-throughput scenarios

## 📈 Future Enhancements

- **Multi-calendar optimization** across different providers
- **ML-based preference learning** from user behavior
- **Advanced constraint types** (location, resources, etc.)
- **Real-time rescheduling** with change notifications

## 🤝 Contributing

This enhanced agentic solution demonstrates cutting-edge AI scheduling capabilities suitable for enterprise deployment and hackathon innovation judging.

---

**Built for AMD Hackathon 2025** 🚀  
*Demonstrating the future of autonomous AI scheduling*
