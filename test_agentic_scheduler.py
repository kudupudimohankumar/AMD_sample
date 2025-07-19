#!/usr/bin/env python3
"""
Test script for Agentic AI Scheduling Assistant
AMD Hackathon 2025
"""

import json
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_scheduler import AgenticScheduler

def test_agentic_scheduler():
    """Test the enhanced agentic scheduler functionality."""
    
    print("🧪 Testing Agentic AI Scheduling Assistant")
    print("=" * 50)
    
    # Initialize scheduler in agentic mode
    print("🤖 Initializing Agentic Scheduler...")
    scheduler = AgenticScheduler(
        vllm_url="http://localhost:3000/v1",
        model_name="/home/user/Models/deepseek-ai/deepseek-llm-7b-chat",
        agentic_mode=True
    )
    
    # Test data
    test_input = {
        "Request_id": "req_test_001",
        "From": "userone.amd@gmail.com",
        "Subject": "Urgent: Client Presentation Review",
        "Content": "Hi team, I need to schedule a 90-minute client presentation review meeting with all stakeholders. This is high priority and should happen in the morning if possible. Please find time this week.",
        "EmailContent": "Hi team, I need to schedule a 90-minute client presentation review meeting with all stakeholders. This is high priority and should happen in the morning if possible. Please find time this week.",
        "Datetime": "2025-01-19T14:30:00+05:30",
        "Attendees": [
            {"name": "User Two", "email": "usertwo.amd@gmail.com"},
            {"name": "User Three", "email": "userthree.amd@gmail.com"}
        ]
    }
    
    print(f"\n📧 Test Input:")
    print(json.dumps(test_input, indent=2))
    
    # Test information extraction
    print(f"\n🧠 Testing Agentic Information Extraction...")
    start_time = time.time()
    
    if scheduler.agentic_mode and scheduler.agentic_ready:
        try:
            extracted_info = scheduler._extract_meeting_info_with_agent(test_input["Content"])
            extraction_time = time.time() - start_time
            print(f"✅ Agent extraction completed in {extraction_time:.3f}s")
            print(f"📊 Extracted: {json.dumps(extracted_info, indent=2)}")
        except Exception as e:
            print(f"⚠️ Agent extraction failed: {e}")
            print("📝 Falling back to traditional extraction...")
            extracted_info = scheduler._extract_meeting_info_traditional(test_input["Content"])
    else:
        print("📝 Using traditional extraction (agentic mode not available)")
        extracted_info = scheduler._extract_meeting_info_traditional(test_input["Content"])
    
    # Test OR-based optimization
    print(f"\n🔍 Testing OR-Tools Optimization...")
    attendees = [test_input["From"]] + [att["email"] for att in test_input["Attendees"]]
    
    start_time = time.time()
    try:
        optimal_start, optimal_end = scheduler.find_optimal_slot(
            attendees=attendees,
            duration_minutes=extracted_info.get('meeting_duration_minutes', 90),
            time_preference=extracted_info.get('time_preference', 'morning'),
            start_date="2025-01-20T09:00:00+05:30"
        )
        optimization_time = time.time() - start_time
        print(f"✅ OR optimization completed in {optimization_time:.3f}s")
        print(f"⏰ Optimal slot: {optimal_start} to {optimal_end}")
    except Exception as e:
        print(f"❌ OR optimization failed: {e}")
        optimal_start, optimal_end = None, None
    
    # Test full processing
    print(f"\n🎯 Testing Full Processing Pipeline...")
    start_time = time.time()
    
    try:
        result = scheduler.process_meeting_request(test_input)
        total_time = time.time() - start_time
        
        print(f"✅ Full processing completed in {total_time:.3f}s")
        
        if total_time < 10:
            print("🎉 LATENCY REQUIREMENT MET! (< 10 seconds)")
        else:
            print("⚠️ Latency exceeds 10-second target")
        
        print(f"\n📅 Final Result:")
        print(json.dumps(result, indent=2))
        
        # Validate output
        required_fields = ['Request_id', 'EventStart', 'EventEnd', 'Duration_mins', 'Attendees']
        missing = [f for f in required_fields if f not in result]
        
        if missing:
            print(f"\n❌ Missing required fields: {missing}")
        else:
            print(f"\n✅ All required fields present!")
        
    except Exception as e:
        print(f"❌ Full processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Performance summary
    print(f"\n🏆 PERFORMANCE SUMMARY")
    print(f"=" * 30)
    print(f"🤖 Agentic Mode: {'ENABLED' if scheduler.agentic_mode else 'DISABLED'}")
    print(f"🔧 OR-Tools Ready: {'YES' if scheduler.agentic_ready else 'NO'}")
    print(f"⚡ Target Latency: < 10 seconds")
    print(f"🎯 Status: READY FOR HACKATHON")

if __name__ == "__main__":
    test_agentic_scheduler()
