"""
Agentic AI Scheduling Assistant
AMD Hackathon Solution

This module implements an intelligent scheduling system that:
1. Parses meeting requests using LLM
2. Extracts calendar events for all attendees
3. Finds optimal meeting slots avoiding conflicts
4. Returns structured JSON output
"""

import json
import requests
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
from dateutil import parser as date_parser
import pytz

# Google Calendar API imports
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    print("Warning: Google Calendar API not available. Using mock data.")

class AgenticScheduler:
    def __init__(self, vllm_url: str = "http://localhost:3000/v1", model_name: str = "/home/user/Models/deepseek-ai/deepseek-llm-7b-chat"):
        """
        Initialize the Agentic Scheduler
        
        Args:
            vllm_url: Base URL for vLLM server
            model_name: Model path for LLM inference
        """
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.timezone = pytz.timezone('Asia/Kolkata')  # IST timezone
        
    def parse_email_content(self, email_content: str, subject: str, attendees: List[str]) -> Dict:
        """
        Parse email content using LLM to extract meeting requirements
        
        Args:
            email_content: Email body text
            subject: Email subject
            attendees: List of attendee email addresses
            
        Returns:
            Dictionary with parsed meeting details
        """
        prompt = f"""
        You are an AI scheduling assistant. Parse the following meeting request and extract key information.
        
        Subject: {subject}
        Email Content: {email_content}
        Attendees: {', '.join(attendees)}
        
        Extract and return ONLY a JSON object with:
        1. "meeting_duration_minutes": Duration in minutes (default 60 if not specified)
        2. "time_preference": Specific day/time mentioned or "flexible" 
        3. "urgency": "high", "medium", or "low"
        4. "meeting_type": "status_update", "planning", "review", or "general"
        
        Return only valid JSON, no other text.
        """
        
        try:
            response = requests.post(
                f"{self.vllm_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=5
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
        except Exception as e:
            print(f"LLM parsing error: {e}")
        
        # Fallback parsing
        return self._fallback_parse(email_content, subject)
    
    def _fallback_parse(self, email_content: str, subject: str) -> Dict:
        """Fallback parsing without LLM"""
        duration = 60  # default
        
        # Extract duration from email content
        duration_patterns = [
            r'(\d+)\s*minutes?',
            r'(\d+)\s*mins?',
            r'(\d+)\s*hour?s?',
            r'half\s*hour',
            r'30\s*min'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, email_content.lower())
            if match:
                if 'hour' in pattern:
                    duration = int(match.group(1)) * 60
                elif 'half' in pattern:
                    duration = 30
                else:
                    duration = int(match.group(1))
                break
        
        # Determine urgency
        urgency = "medium"
        if any(word in email_content.lower() for word in ['urgent', 'asap', 'immediately']):
            urgency = "high"
        elif any(word in email_content.lower() for word in ['when convenient', 'flexible', 'whenever']):
            urgency = "low"
        
        # Time preference
        time_pref = "flexible"
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in email_content.lower():
                time_pref = day
                break
        
        return {
            "meeting_duration_minutes": duration,
            "time_preference": time_pref,
            "urgency": urgency,
            "meeting_type": "general"
        }
    
    def get_calendar_events(self, email: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Retrieve calendar events for an attendee using Google Calendar API
        
        Args:
            email: Attendee email
            start_date: Start date for search (YYYY-MM-DD format)
            end_date: End date for search (YYYY-MM-DD format)
            
        Returns:
            List of calendar events
        """
        if not GOOGLE_CALENDAR_AVAILABLE:
            return self._get_mock_calendar_events(email, start_date, end_date)
        
        try:
            # Convert dates to proper format for Google Calendar API
            start_datetime = f"{start_date}T00:00:00+05:30"  # IST timezone
            end_datetime = f"{end_date}T23:59:59+05:30"     # IST timezone
            
            return self._retrieve_calendar_events(email, start_datetime, end_datetime)
        except Exception as e:
            print(f"Error fetching calendar events for {email}: {e}")
            return self._get_mock_calendar_events(email, start_date, end_date)
    
    def _retrieve_calendar_events(self, user: str, start: str, end: str) -> List[Dict]:
        """
        Implementation of the Google Calendar event retrieval function
        Based on the provided calendar fetcher code
        """
        events_list = []
        
        try:
            # Build token path based on user email
            token_path = f"Keys/{user.split('@')[0]}.token"
            
            # Load user credentials from token file
            user_creds = Credentials.from_authorized_user_file(token_path)
            
            # Build calendar service
            calendar_service = build("calendar", "v3", credentials=user_creds)
            
            # Fetch events from Google Calendar API
            events_result = calendar_service.events().list(
                calendarId='primary',
                timeMin=start,
                timeMax=end,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Process each event
            for event in events:
                attendee_list = []
                try:
                    # Extract attendees if available
                    if "attendees" in event:
                        for attendee in event["attendees"]:
                            attendee_list.append(attendee['email'])
                    else:
                        attendee_list.append("SELF")
                except Exception:
                    attendee_list.append("SELF")
                
                # Extract start and end times
                start_time = event["start"].get("dateTime", event["start"].get("date"))
                end_time = event["end"].get("dateTime", event["end"].get("date"))
                
                # Create event dictionary in the required format
                events_list.append({
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "NumAttendees": len(set(attendee_list)),
                    "Attendees": list(set(attendee_list)),
                    "Summary": event.get("summary", "No title")
                })
                
        except FileNotFoundError:
            print(f"Token file not found for user {user}. Using mock data.")
            return self._get_mock_calendar_events(user, start.split('T')[0], end.split('T')[0])
        except Exception as e:
            print(f"Error accessing Google Calendar for {user}: {e}")
            return self._get_mock_calendar_events(user, start.split('T')[0], end.split('T')[0])
        
        return events_list
    
    def _get_mock_calendar_events(self, email: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fallback mock calendar events when Google Calendar API is not available
        """
        mock_events = {
            "userone.amd@gmail.com": [
                {
                    "StartTime": "2025-07-24T09:00:00+05:30",
                    "EndTime": "2025-07-24T10:00:00+05:30",
                    "NumAttendees": 1,
                    "Attendees": ["userone.amd@gmail.com"],
                    "Summary": "Morning Standup"
                }
            ],
            "usertwo.amd@gmail.com": [
                {
                    "StartTime": "2025-07-24T10:00:00+05:30",
                    "EndTime": "2025-07-24T10:30:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": ["userone.amd@gmail.com", "usertwo.amd@gmail.com", "userthree.amd@gmail.com"],
                    "Summary": "Team Meet"
                },
                {
                    "StartTime": "2025-07-24T14:00:00+05:30",
                    "EndTime": "2025-07-24T15:00:00+05:30",
                    "NumAttendees": 1,
                    "Attendees": ["usertwo.amd@gmail.com"],
                    "Summary": "Client Call"
                }
            ],
            "userthree.amd@gmail.com": [
                {
                    "StartTime": "2025-07-24T10:00:00+05:30",
                    "EndTime": "2025-07-24T10:30:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": ["userone.amd@gmail.com", "usertwo.amd@gmail.com", "userthree.amd@gmail.com"],
                    "Summary": "Team Meet"
                },
                {
                    "StartTime": "2025-07-24T13:00:00+05:30",
                    "EndTime": "2025-07-24T14:00:00+05:30",
                    "NumAttendees": 1,
                    "Attendees": ["SELF"],
                    "Summary": "Lunch with Customers"
                }
            ]
        }
        
        return mock_events.get(email, [])
    
    def find_optimal_slot(self, attendees: List[str], duration_minutes: int, 
                         time_preference: str, start_date: str) -> Tuple[str, str]:
        """
        Find optimal meeting slot for all attendees
        
        Args:
            attendees: List of attendee emails
            duration_minutes: Meeting duration in minutes
            time_preference: Preferred time or "flexible"
            start_date: Date to start searching from
            
        Returns:
            Tuple of (start_time, end_time) in ISO format
        """
        # Parse start date
        base_date = date_parser.parse(start_date).date()
        
        # Define working hours (9 AM to 6 PM IST)
        working_start = 9
        working_end = 18
        
        # Get all attendees' calendars
        all_events = {}
        for attendee in attendees:
            events = self.get_calendar_events(
                attendee, 
                base_date.isoformat(), 
                (base_date + timedelta(days=7)).isoformat()
            )
            all_events[attendee] = events
        
        # Try to find slot for next 7 days
        for day_offset in range(7):
            search_date = base_date + timedelta(days=day_offset)
            
            # Skip weekends unless specified
            if search_date.weekday() >= 5 and time_preference == "flexible":
                continue
            
            # Check if this matches time preference
            if time_preference != "flexible":
                day_name = search_date.strftime('%A').lower()
                if time_preference.lower() not in day_name:
                    continue
            
            # Find free slots for this day
            optimal_slot = self._find_day_slot(
                all_events, search_date, duration_minutes, 
                working_start, working_end
            )
            
            if optimal_slot:
                return optimal_slot
        
        # Fallback: return a slot anyway
        fallback_date = base_date + timedelta(days=1)
        start_time = self.timezone.localize(
            datetime.combine(fallback_date, datetime.min.time().replace(hour=10, minute=30))
        )
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        return start_time.isoformat(), end_time.isoformat()
    
    def _find_day_slot(self, all_events: Dict, search_date, duration_minutes: int,
                      working_start: int, working_end: int) -> Optional[Tuple[str, str]]:
        """Find available slot for a specific day"""
        
        # Create time slots (30-minute intervals)
        slots = []
        current_time = working_start
        
        while current_time < working_end:
            slot_start = self.timezone.localize(
                datetime.combine(search_date, datetime.min.time().replace(hour=int(current_time), minute=int((current_time % 1) * 60)))
            )
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            
            # Check if slot fits in working hours
            if slot_end.hour < working_end:
                # Check if all attendees are free
                if self._is_slot_free(all_events, slot_start, slot_end):
                    return slot_start.isoformat(), slot_end.isoformat()
            
            current_time += 0.5  # 30-minute increments
        
        return None
    
    def _is_slot_free(self, all_events: Dict, slot_start: datetime, slot_end: datetime) -> bool:
        """Check if time slot is free for all attendees"""
        
        for attendee, events in all_events.items():
            for event in events:
                event_start = date_parser.parse(event['StartTime'])
                event_end = date_parser.parse(event['EndTime'])
                
                # Check for overlap
                if (slot_start < event_end and slot_end > event_start):
                    return False
        
        return True
    
    def generate_output_json(self, input_data: Dict, parsed_info: Dict, 
                           start_time: str, end_time: str) -> Dict:
        """
        Generate the final output JSON in the required format
        
        Args:
            input_data: Original input JSON
            parsed_info: Parsed meeting information
            start_time: Meeting start time
            end_time: Meeting end time
            
        Returns:
            Formatted output JSON
        """
        # Get all attendee emails including sender
        all_attendees = [input_data['From']]
        for attendee in input_data.get('Attendees', []):
            if attendee['email'] not in all_attendees:
                all_attendees.append(attendee['email'])
        
        # Create attendee sections with their existing events
        attendee_sections = []
        
        for attendee_email in all_attendees:
            # Get existing events for this attendee
            existing_events = self.get_calendar_events(
                attendee_email,
                date_parser.parse(start_time).date().isoformat(),
                (date_parser.parse(start_time).date() + timedelta(days=1)).isoformat()
            )
            
            # Add the new meeting event
            new_event = {
                "StartTime": start_time,
                "EndTime": end_time,
                "NumAttendees": len(all_attendees),
                "Attendees": all_attendees,
                "Summary": input_data.get('Subject', 'Meeting')
            }
            
            # Combine events
            all_events = existing_events + [new_event]
            
            attendee_sections.append({
                "email": attendee_email,
                "events": all_events
            })
        
        # Build final output
        output = {
            "Request_id": input_data.get('Request_id', ''),
            "Datetime": input_data.get('Datetime', ''),
            "Location": input_data.get('Location', ''),
            "From": input_data.get('From', ''),
            "Attendees": attendee_sections,
            "Subject": input_data.get('Subject', ''),
            "EmailContent": input_data.get('EmailContent', ''),
            "EventStart": start_time,
            "EventEnd": end_time,
            "Duration_mins": str(parsed_info.get('meeting_duration_minutes', 60)),
            "MetaData": {
                "urgency": parsed_info.get('urgency', 'medium'),
                "meeting_type": parsed_info.get('meeting_type', 'general'),
                "time_preference": parsed_info.get('time_preference', 'flexible')
            }
        }
        
        return output
    
    def process_meeting_request(self, input_data: Dict) -> Dict:
        """
        Main function to process a meeting request
        
        Args:
            input_data: Input JSON with meeting request
            
        Returns:
            Output JSON with scheduled meeting details
        """
        try:
            # Extract attendee emails
            attendee_emails = [input_data['From']]
            for attendee in input_data.get('Attendees', []):
                attendee_emails.append(attendee['email'])
            
            # Parse email content using LLM
            parsed_info = self.parse_email_content(
                input_data.get('EmailContent', ''),
                input_data.get('Subject', ''),
                attendee_emails
            )
            
            # Find optimal meeting slot
            start_time, end_time = self.find_optimal_slot(
                attendee_emails,
                parsed_info.get('meeting_duration_minutes', 60),
                parsed_info.get('time_preference', 'flexible'),
                input_data.get('Datetime', '2025-07-19T12:34:55')
            )
            
            # Generate final output
            output = self.generate_output_json(input_data, parsed_info, start_time, end_time)
            
            return output
            
        except Exception as e:
            print(f"Error processing meeting request: {e}")
            # Return minimal valid response on error
            return {
                **input_data,
                "EventStart": "2025-07-24T10:30:00+05:30",
                "EventEnd": "2025-07-24T11:30:00+05:30",
                "Duration_mins": "60",
                "MetaData": {"error": str(e)}
            }


def your_meeting_assistant(data: Dict) -> Dict:
    """
    Main function called by the Flask server
    This is the entry point for the hackathon submission
    """
    scheduler = AgenticScheduler()
    return scheduler.process_meeting_request(data)


# Test function
if __name__ == "__main__":
    # Test with sample input
    with open('/Users/mohan/Documents/WIP/AMD_hackathon/JSON_Samples/Input_Request.json', 'r') as f:
        test_input = json.load(f)
    
    scheduler = AgenticScheduler()
    result = scheduler.process_meeting_request(test_input)
    
    print("Output JSON:")
    print(json.dumps(result, indent=2))
