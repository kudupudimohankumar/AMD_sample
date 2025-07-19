#!/usr/bin/env python3
"""
Test script for the Agentic AI Scheduling Assistant
This script tests the implementation with the sample input
"""

import json
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_scheduler import AgenticScheduler

def test_scheduler():
    """Test the scheduler with sample input"""
    
    # Load test input
    with open('JSON_Samples/Input_Request.json', 'r') as f:
        test_input = json.load(f)
    
    print("=" * 60)
    print("AGENTIC AI SCHEDULING ASSISTANT - TEST")
    print("=" * 60)
    
    print("\n📧 INPUT REQUEST:")
    print(json.dumps(test_input, indent=2))
    
    # Initialize scheduler
    scheduler = AgenticScheduler()
    
    print("\n🤖 PROCESSING WITH AGENTIC AI...")
    
    # Process the request
    result = scheduler.process_meeting_request(test_input)
    
    print("\n📅 GENERATED SCHEDULE:")
    print(json.dumps(result, indent=2))
    
    # Validate output format
    print("\n✅ VALIDATION:")
    required_fields = [
        'Request_id', 'Datetime', 'Location', 'From', 
        'Attendees', 'Subject', 'EmailContent', 
        'EventStart', 'EventEnd', 'Duration_mins', 'MetaData'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in result:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    else:
        print("✅ All required fields present")
    
    # Check attendee structure
    if 'Attendees' in result and isinstance(result['Attendees'], list):
        for i, attendee in enumerate(result['Attendees']):
            if 'email' not in attendee or 'events' not in attendee:
                print(f"❌ Attendee {i} missing required fields")
                return False
        print("✅ Attendee structure valid")
    
    print("\n🎯 SUCCESS: Implementation passed all tests!")
    return True

def benchmark_performance():
    """Benchmark the performance"""
    import time
    
    with open('JSON_Samples/Input_Request.json', 'r') as f:
        test_input = json.load(f)
    
    scheduler = AgenticScheduler()
    
    print("\n⏱️  PERFORMANCE BENCHMARK:")
    
    # Run multiple iterations
    times = []
    for i in range(5):
        start_time = time.time()
        result = scheduler.process_meeting_request(test_input)
        end_time = time.time()
        
        duration = end_time - start_time
        times.append(duration)
        print(f"Run {i+1}: {duration:.3f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average processing time: {avg_time:.3f} seconds")
    
    if avg_time < 10.0:
        print("✅ Meets latency requirement (<10 seconds)")
    else:
        print("❌ Exceeds latency requirement (>10 seconds)")
    
    return avg_time

if __name__ == "__main__":
    print("Starting Agentic AI Scheduling Assistant Test...\n")
    
    # Test functionality
    success = test_scheduler()
    
    if success:
        # Benchmark performance
        benchmark_performance()
        
        print("\n" + "=" * 60)
        print("🏆 READY FOR HACKATHON SUBMISSION!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start vLLM server with DeepSeek model")
        print("2. Run the Submission.ipynb notebook")
        print("3. Test with external requests on port 5000")
        print("\nGood luck with the hackathon! 🚀")
    else:
        print("\n❌ Test failed. Please check the implementation.")
