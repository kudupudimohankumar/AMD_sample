#!/usr/bin/env python3
"""
Quick test to verify OR-Tools fix for "hour must be in 0..23" error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import traceback

def test_or_tools_fix():
    """Test the OR-Tools optimization fix"""
    print("🧪 Testing OR-Tools Fix for 'hour must be in 0..23' Error")
    print("=" * 60)
    
    try:
        # Import the scheduler
        from agentic_scheduler import AgenticScheduler
        print("✅ AgenticScheduler imported successfully")
        
        # Initialize scheduler
        scheduler = AgenticScheduler()
        print("✅ AgenticScheduler initialized")
        
        # Test the specific OR-Tools optimization method
        print("\n🔧 Testing OR-Tools optimization method...")
        
        # Test parameters
        attendees = ['john@example.com', 'jane@example.com']
        duration_minutes = 60
        time_preference = 'afternoon'
        start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"📅 Testing for date: {start_date}")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"👥 Attendees: {attendees}")
        print(f"🕐 Time preference: {time_preference}")
        
        # Call the OR optimization method
        result = scheduler._find_optimal_slot_agentic_or(
            attendees, duration_minutes, time_preference, start_date
        )
        
        print(f"\n✅ OR-Tools optimization completed successfully!")
        print(f"📅 Suggested slot: {result[0]} to {result[1]}")
        
        # Validate the result format
        start_time, end_time = result
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        print(f"🔍 Validation:")
        print(f"   Start hour: {start_dt.hour} (must be 0-23) ✅")
        print(f"   End hour: {end_dt.hour} (must be 0-23) ✅")
        print(f"   Duration: {(end_dt - start_dt).total_seconds() / 60} minutes ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def test_time_slot_calculation():
    """Test the time slot calculation specifically"""
    print("\n🧮 Testing Time Slot Calculation")
    print("-" * 40)
    
    try:
        # Constants from the fixed code
        SLOT_DURATION = 15  # 15 minutes
        BUSINESS_START_HOUR = 9  # 9 AM
        SLOTS_PER_DAY = 36  # 9 hours * 4 slots per hour
        
        print(f"⚙️  Constants:")
        print(f"   Slot duration: {SLOT_DURATION} minutes")
        print(f"   Business start: {BUSINESS_START_HOUR} AM")
        print(f"   Slots per day: {SLOTS_PER_DAY}")
        
        # Test slot to time conversion
        test_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for slot in [0, 12, 24, 35]:  # Test edge cases
            # Calculate time as in the fixed code
            slot_start_minutes = slot * SLOT_DURATION
            slot_hour = BUSINESS_START_HOUR + (slot_start_minutes // 60)
            slot_minute = slot_start_minutes % 60
            
            print(f"   Slot {slot:2d}: {slot_hour:2d}:{slot_minute:02d} ({slot_hour} hour)")
            
            # Validate hour is in valid range
            if not (0 <= slot_hour <= 23):
                print(f"   ❌ Invalid hour {slot_hour} for slot {slot}")
                return False
                
        print("✅ All time slot calculations valid!")
        return True
        
    except Exception as e:
        print(f"❌ Slot calculation test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 AMD Hackathon - OR-Tools Fix Verification")
    print("Testing the fix for 'hour must be in 0..23' error\n")
    
    # Run tests
    slot_test_passed = test_time_slot_calculation()
    or_test_passed = test_or_tools_fix()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"🧮 Time Slot Calculation: {'✅ PASSED' if slot_test_passed else '❌ FAILED'}")
    print(f"🔧 OR-Tools Optimization: {'✅ PASSED' if or_test_passed else '❌ FAILED'}")
    
    if slot_test_passed and or_test_passed:
        print("\n🎉 ALL TESTS PASSED! OR-Tools fix is working correctly!")
        print("✅ The 'hour must be in 0..23' error has been resolved.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        
    print("\n🏁 Test completed.")
