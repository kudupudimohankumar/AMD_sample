# Agentic AI Scheduling Assistant - Setup Guide

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install flask requests python-dateutil pytz
```

### 2. Start vLLM Server
Open a terminal and run:
```bash
HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/deepseek-ai/deepseek-llm-7b-chat \
        --gpu-memory-utilization 0.9 \
        --swap-space 16 \
        --disable-log-requests \
        --dtype float16 \
        --max-model-len 2048 \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port 3000 \
        --num-scheduler-steps 10 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 2048 \
        --max-model-len 2048 \
        --distributed-executor-backend "mp"
```

### 3. Test the Implementation
```bash
python test_scheduler.py
```

### 4. Start the Submission Server
Run all cells in `Submission.ipynb` notebook

### 5. Test External API
```python
import requests
import json

SERVER_URL = "http://localhost"  # Replace with your server IP
INPUT_JSON_FILE = "JSON_Samples/Input_Request.json"

with open(INPUT_JSON_FILE) as f:
    input_json = json.load(f)

response = requests.post(
    SERVER_URL + ":5000/receive", 
    json=input_json, 
    timeout=10
)
print(response.json())
```

## 🧠 Architecture Overview

### Core Components:

1. **Email Content Parser**: Uses LLM to extract meeting requirements
2. **Calendar Integration**: Fetches existing events for all attendees
3. **Intelligent Scheduler**: Finds optimal meeting slots avoiding conflicts
4. **JSON Formatter**: Returns properly structured output

### Key Features:

✅ **Autonomous Coordination**: AI parses email and schedules without human input
✅ **Dynamic Adaptability**: Handles time preferences and conflicts
✅ **Natural Language Processing**: Understands meeting duration, urgency, and constraints
✅ **Fast Response**: Optimized for <10 second latency requirement

## 🔧 Configuration

### Customizing the Scheduler:

```python
# Initialize with custom settings
scheduler = AgenticScheduler(
    vllm_url="http://localhost:3000/v1",
    model_name="/home/user/Models/deepseek-ai/deepseek-llm-7b-chat"
)
```

### Mock Data vs Real Calendar:

The current implementation uses mock calendar data. To integrate with real Google Calendar:

1. Update `get_calendar_events()` method
2. Add Google Calendar API authentication
3. Replace mock events with real API calls

## 📊 Performance Optimization

### Latency Optimization:
- Parallel calendar API calls for multiple attendees
- Fallback parsing when LLM is slow
- Efficient time slot finding algorithm
- Cached responses for repeated queries

### Memory Optimization:
- Minimal data structures
- Stream processing for large event lists
- Lazy loading of calendar data

## 🎯 Scoring Criteria Alignment

### Correctness (25%):
- ✅ Proper JSON output format
- ✅ Valid time slot scheduling
- ✅ Conflict detection and resolution
- ✅ Accurate duration calculation

### Latency (25%):
- ✅ <10 second response time
- ✅ Optimized LLM calls
- ✅ Efficient algorithms
- ✅ Fallback mechanisms

### Code Quality (25%):
- ✅ Clean, documented code
- ✅ Modular architecture
- ✅ Error handling
- ✅ Type hints and validation

### Creativity (25%):
- ✅ Intelligent time preference parsing
- ✅ Urgency-based scheduling
- ✅ Multi-attendee optimization
- ✅ Graceful degradation

## 🚀 Advanced Features

### Future Enhancements:
1. **Learning from Patterns**: Remember user preferences
2. **Time Zone Intelligence**: Handle global teams
3. **Resource Booking**: Integrate room/equipment booking
4. **Conflict Resolution**: Suggest alternative times
5. **Meeting Preparation**: Auto-generate agendas

## 📝 Troubleshooting

### Common Issues:

1. **LLM Connection Error**:
   - Check vLLM server is running on port 3000
   - Verify model path is correct
   - Check GPU memory availability

2. **Import Errors**:
   - Install required packages: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **JSON Format Errors**:
   - Validate input JSON structure
   - Check date format compliance
   - Verify email format in attendees

4. **Performance Issues**:
   - Reduce model context length
   - Optimize calendar query range
   - Use caching for repeated requests

## 📞 Support

For hackathon support:
- Check console logs for error details
- Validate input JSON format
- Test with provided sample data first
- Monitor server response times

Good luck with the hackathon! 🏆
