# ------------------------------------------------------------------
#  NEW IMPORTS for the OR-Tools engine
# ------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from ortools.sat.python import cp_model
import datetime as _dt
from dateutil import parser as _dp

import json
import requests
from datetime import datetime, timedelta, time
import re
from typing import Dict, List, Tuple, Optional, Any
from dateutil import parser as date_parser
import pytz

IST = pytz.timezone('Asia/Kolkata')
SEARCH_D = 7                 # look ahead 7 calendar days
DUR_STEP = 15                # granularity for candidate slots (minutes)
WORKDAY = (9, 18)            # working hours 09:00–18:00 IST


# ------------------------------------------------------------------
#  OR-TOOLS SLOT FINDER
# ------------------------------------------------------------------
def _events(email: str, start_date: str, end_date: str) -> List[dict]:
    """
    Thin wrapper so the OR-Tools layer can reuse the existing
    AgenticScheduler().get_calendar_events method.
    """
    return AgenticScheduler().get_calendar_events(email, start_date, end_date)


def _busy(events: List[dict]) -> List[Tuple[_dt.datetime, _dt.datetime]]:
    """
    Convert Google-formatted events into simple datetime tuples.
    """
    busy = []
    for ev in events:
        start = _dp.parse(ev["StartTime"]).astimezone(IST)
        end   = _dp.parse(ev["EndTime"]).astimezone(IST)
        busy.append((start, end))
    return busy


def _optimal(attendees: List[str], dur: int, start_date: str) -> Tuple[str, str]:
    """
    Use OR-Tools CP-SAT to pick the earliest *optimal* slot
    that respects everyone’s existing calendar.
    Falls back to greedy if the solver times out or no model found.
    """
    base = _dt.datetime.combine(_dp.parse(start_date).date(), _dt.time(9, 0, tzinfo=IST))
    duration = _dt.timedelta(minutes=dur)
    busy = []

    # --- Fetch all calendars in parallel -----------------------------
    with ThreadPoolExecutor() as ex:
        for ev_list in ex.map(
            lambda u: _events(u, base.date().isoformat(),
                              (base + _dt.timedelta(days=SEARCH_D)).date().isoformat()),
            attendees
        ):
            busy.extend(_busy(ev_list))

    # --- Build candidate slots (Mon–Fri, 09:00–18:00) ---------------
    slots = []
    cur = base
    while cur + duration <= base + _dt.timedelta(days=SEARCH_D):
        if cur.weekday() < 5 and WORKDAY[0] <= cur.hour < WORKDAY[1]:
            slots.append((cur, cur + duration))
        cur += _dt.timedelta(minutes=DUR_STEP)

    # --- CP-SAT model ----------------------------------------------
    if slots:
        mdl = cp_model.CpModel()
        x = [mdl.NewBoolVar(f's{i}') for i in range(len(slots))]
        mdl.Add(sum(x) == 1)               # choose exactly one slot

        for i, (s, e) in enumerate(slots):
            # forbid if it overlaps any busy interval
            if any(s < b[1] and e > b[0] for b in busy):
                mdl.Add(x[i] == 0)

        # objective: earliest start time
        mdl.Minimize(sum(x[i] * int(s.timestamp()) for i, (s, _) in enumerate(slots)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.5
        if solver.Solve(mdl) == cp_model.OPTIMAL:
            idx = next(i for i, b in enumerate(x) if solver.Value(b))
            return slots[idx][0].isoformat(), slots[idx][1].isoformat()

    # --- Greedy fallback -------------------------------------------
    for s, e in slots:
        if not any(s < b[1] and e > b[0] for b in busy):
            return s.isoformat(), e.isoformat()

    # --- Ultimate fallback -----------------------------------------
    fb = base + _dt.timedelta(days=1, hours=1, minutes=30)
    return fb.isoformat(), (fb + duration).isoformat()



"""
Agentic AI Scheduling Assistant
AMD Hackathon Solution

This module implements an intelligent scheduling system that:
1. Parses meeting requests using LLM
2. Extracts calendar events for all attendees
3. Finds optimal meeting slots avoiding conflicts
4. Returns structured JSON output
"""



# Google Calendar API imports
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    print("Warning: Google Calendar API not available. Using mock data.")

class AgenticScheduler:
    def __init__(self, vllm_url: str = "http://localhost:3000/v1", 
                 model_name: str = "/home/user/Models/deepseek-ai/deepseek-llm-7b-chat",
                 agentic_mode: bool = True):
        """
        Initialize the Agentic Scheduler
        
        Args:
            vllm_url: Base URL for vLLM server
            model_name: Model path for LLM inference
            agentic_mode: Enable advanced agentic AI capabilities
        """
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.timezone = pytz.timezone('Asia/Kolkata')  # IST timezone
        self.agentic_mode = agentic_mode
        
        if self.agentic_mode:
            print("🤖 Agentic AI Mode: ENABLED")
            print("✨ Using advanced AI agents with MCP and OR optimization")
            self._init_agentic_components()
    
    def _init_agentic_components(self):
        """Initialize agentic AI components."""
        try:
            from pydantic_ai.models.openai import OpenAIModel
            from pydantic_ai.providers.openai import OpenAIProvider
            from pydantic_ai import Agent, Tool
            
            # Configure AI provider for agentic operations
            self.provider = OpenAIProvider(
                base_url=self.vllm_url,
                api_key="amd-hackathon-2025",
            )
            
            self.agent_model = OpenAIModel(self.model_name, provider=self.provider)
            self.agentic_ready = True
            print("🧠 Agentic components initialized successfully!")
            
        except ImportError:
            print("⚠️ Pydantic AI not available, falling back to traditional mode")
            self.agentic_ready = False
            self.agentic_mode = False
        
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
    
    def _extract_meeting_info_with_agent(self, email_content: str) -> Dict[str, Any]:
        """
        Extract meeting information using Agentic AI with MCP tools.
        
        Args:
            email_content: Raw email content
            
        Returns:
            Dict containing extracted meeting parameters
        """
        if not self.agentic_mode or not self.agentic_ready:
            return self._extract_meeting_info_traditional(email_content)
        
        try:
            from pydantic_ai import Agent
            from pydantic import BaseModel
            from typing import Optional
            
            class MeetingInfo(BaseModel):
                meeting_duration_minutes: int
                time_preference: str  # morning, afternoon, evening, anytime
                urgency: str  # high, medium, low
                meeting_type: str  # team_meeting, client_call, interview, presentation
                preferred_date: Optional[str] = None
                attendees_count: Optional[int] = None
                location_preference: Optional[str] = None
                recurring: bool = False
            
            # Create specialized extraction agent
            extraction_agent = Agent(
                model=self.agent_model,
                result_type=MeetingInfo,
                system_prompt="""You are an expert meeting information extraction agent.

Extract these key parameters from email content:
1. meeting_duration_minutes: Estimate duration (default 60 if not specified)
2. time_preference: morning (6-12), afternoon (12-17), evening (17-21), anytime
3. urgency: high (ASAP, urgent), medium (this week), low (flexible)
4. meeting_type: team_meeting, client_call, interview, presentation, other

Analyze the context, tone, and explicit requirements carefully.
Be precise and practical in your extractions."""
            )
            
            # Run extraction with the agent
            result = extraction_agent.run_sync(
                f"Extract meeting information from this email:\n\n{email_content}"
            )
            
            extracted_info = result.data.model_dump()
            print(f"🤖 Agent extracted: {extracted_info}")
            return extracted_info
            
        except Exception as e:
            print(f"⚠️ Agent extraction failed: {e}")
            return self._extract_meeting_info_traditional(email_content)
    
    def _extract_meeting_info_traditional(self, email_content: str) -> Dict[str, Any]:
        """Traditional LLM-based extraction as fallback."""
        # Original extraction logic
        prompt = f"""Extract meeting information from this email and return a JSON object:

Email: {email_content}

Return JSON with these fields:
- meeting_duration_minutes (int): Duration in minutes, default 60
- time_preference (str): "morning", "afternoon", "evening", or "anytime"  
- urgency (str): "high", "medium", or "low"
- meeting_type (str): "team_meeting", "client_call", "interview", "presentation", or "other"

JSON:"""

        try:
            response = requests.post(f"{self.vllm_url}/completions", 
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['text'].strip()
                # Parse JSON from response
                import json
                try:
                    meeting_info = json.loads(result)
                    return meeting_info
                except:
                    # Fallback to manual parsing
                    pass
        except:
            pass
        
        # Ultimate fallback - basic extraction
        return {
            "meeting_duration_minutes": 60,
            "time_preference": "anytime",
            "urgency": "medium", 
            "meeting_type": "other"
        }
    
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

    def find_optimal_slot(self,
                          attendees: List[str],
                          duration_minutes: int,
                          time_preference: str,
                          start_date: str) -> Tuple[str, str]:
        """
        Find optimal meeting slot using advanced OR-based optimization with agentic enhancements.
        
        Args:
            attendees: List of attendee email addresses
            duration_minutes: Meeting duration in minutes
            time_preference: Time preference (morning, afternoon, evening, anytime)
            start_date: Start date for search
            
        Returns:
            Tuple of (start_time, end_time) in ISO format
        """
        if self.agentic_mode and self.agentic_ready:
            return self._find_optimal_slot_agentic_or(attendees, duration_minutes, time_preference, start_date)
        else:
            return self._find_optimal_slot_traditional(attendees, duration_minutes, time_preference, start_date)
    
    def _find_optimal_slot_agentic_or(self, attendees: List[str], duration_minutes: int, 
                                     time_preference: str, start_date: str) -> Tuple[str, str]:
        """
        Advanced OR-Tools + Agentic AI optimization for meeting scheduling.
        Uses constraint programming, heuristic algorithms, and AI reasoning.
        """
        try:
            print("🧠 Using Agentic OR-based optimization...")
            
            # Step 1: Collect calendar data for all attendees
            all_events = {}
            search_date = date_parser.parse(start_date).date()
            
            for attendee in attendees:
                events = self.get_calendar_events(
                    attendee, 
                    start_date, 
                    (search_date + timedelta(days=7)).isoformat()
                )
                all_events[attendee] = events
            
            # Step 2: Set up OR-Tools Constraint Programming Model
            model = cp_model.CpModel()
            solver = cp_model.CpSolver()
            
            # Define time slots (15-minute granularity for precision)
            SLOT_DURATION = 15  # minutes
            SLOTS_PER_DAY = (24 * 60) // SLOT_DURATION  # 96 slots per day
            DAYS_TO_SEARCH = 7
            
            # Time preference mapping
            preference_weights = self._get_time_preference_weights(time_preference)
            
            # Step 3: Create decision variables
            # meeting_slot[day][slot] = 1 if meeting starts at this slot on this day
            meeting_slot = {}
            for day in range(DAYS_TO_SEARCH):
                meeting_slot[day] = {}
                for slot in range(SLOTS_PER_DAY):
                    meeting_slot[day][slot] = model.NewBoolVar(f'meeting_d{day}_s{slot}')
            
            # Step 4: Add constraints
            # Constraint 1: Exactly one meeting time must be selected
            meeting_vars = []
            for day in range(DAYS_TO_SEARCH):
                for slot in range(SLOTS_PER_DAY):
                    meeting_vars.append(meeting_slot[day][slot])
            model.Add(sum(meeting_vars) == 1)
            
            # Constraint 2: No conflicts with existing meetings
            slots_needed = (duration_minutes + SLOT_DURATION - 1) // SLOT_DURATION
            
            for day in range(DAYS_TO_SEARCH):
                current_date = search_date + timedelta(days=day)
                
                # Skip weekends (heuristic optimization)
                if current_date.weekday() >= 5:
                    for slot in range(SLOTS_PER_DAY):
                        model.Add(meeting_slot[day][slot] == 0)
                    continue
                
                for slot in range(SLOTS_PER_DAY - slots_needed + 1):
                    # Check if this slot conflicts with any attendee's calendar
                    has_conflict = self._check_slot_conflicts(
                        all_events, current_date, slot, slots_needed, SLOT_DURATION
                    )
                    
                    if has_conflict:
                        model.Add(meeting_slot[day][slot] == 0)
            
            # Constraint 3: Business hours only (9 AM to 6 PM)
            business_start_slot = (9 * 60) // SLOT_DURATION  # 9 AM
            business_end_slot = (18 * 60) // SLOT_DURATION   # 6 PM
            
            for day in range(DAYS_TO_SEARCH):
                for slot in range(SLOTS_PER_DAY):
                    if slot < business_start_slot or slot >= business_end_slot - slots_needed:
                        model.Add(meeting_slot[day][slot] == 0)
            
            # Step 5: Define objective function (maximize preference score)
            objective_terms = []
            for day in range(DAYS_TO_SEARCH):
                for slot in range(SLOTS_PER_DAY):
                    weight = preference_weights.get(slot, 1)
                    # Prefer earlier days (urgency factor)
                    day_weight = max(1, DAYS_TO_SEARCH - day)
                    objective_terms.append(meeting_slot[day][slot] * weight * day_weight)
            
            model.Maximize(sum(objective_terms))
            
            # Step 6: Solve with time limit
            solver.parameters.max_time_in_seconds = 2.0  # Keep under latency requirement
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Extract solution
                for day in range(DAYS_TO_SEARCH):
                    for slot in range(SLOTS_PER_DAY):
                        if solver.Value(meeting_slot[day][slot]) == 1:
                            # Convert back to datetime
                            selected_date = search_date + timedelta(days=day)
                            start_minutes = slot * SLOT_DURATION
                            start_hour = start_minutes // 60
                            start_min = start_minutes % 60
                            
                            start_time = self.timezone.localize(
                                datetime.combine(selected_date, time(start_hour, start_min))
                            )
                            end_time = start_time + timedelta(minutes=duration_minutes)
                            
                            print(f"✅ OR-Tools found optimal slot: {start_time}")
                            return start_time.isoformat(), end_time.isoformat()
            
            print("⚠️ OR-Tools couldn't find optimal solution, using heuristic fallback")
            return self._heuristic_fallback(attendees, duration_minutes, time_preference, start_date)
            
        except Exception as e:
            print(f"❌ OR optimization failed: {e}")
            return self._find_optimal_slot_traditional(attendees, duration_minutes, time_preference, start_date)
    
    def _get_time_preference_weights(self, time_preference: str) -> Dict[int, int]:
        """Get optimization weights based on time preference."""
        weights = {}
        SLOT_DURATION = 15
        
        if time_preference == "morning":
            # Weight morning slots (9 AM - 12 PM) higher
            for hour in range(9, 12):
                for min_slot in range(0, 60, SLOT_DURATION):
                    slot = ((hour * 60) + min_slot) // SLOT_DURATION
                    weights[slot] = 10
        elif time_preference == "afternoon":
            # Weight afternoon slots (12 PM - 5 PM) higher
            for hour in range(12, 17):
                for min_slot in range(0, 60, SLOT_DURATION):
                    slot = ((hour * 60) + min_slot) // SLOT_DURATION
                    weights[slot] = 10
        elif time_preference == "evening":
            # Weight late afternoon/early evening (4 PM - 6 PM) higher
            for hour in range(16, 18):
                for min_slot in range(0, 60, SLOT_DURATION):
                    slot = ((hour * 60) + min_slot) // SLOT_DURATION
                    weights[slot] = 10
        
        return weights
    
    def _check_slot_conflicts(self, all_events: Dict, date: datetime, 
                             start_slot: int, slots_needed: int, slot_duration: int) -> bool:
        """Check if a time slot conflicts with existing meetings."""
        start_minutes = start_slot * slot_duration
        end_minutes = (start_slot + slots_needed) * slot_duration
        
        slot_start = self.timezone.localize(
            datetime.combine(date, time(start_minutes // 60, start_minutes % 60))
        )
        slot_end = self.timezone.localize(
            datetime.combine(date, time(end_minutes // 60, end_minutes % 60))
        )
        
        for attendee, events in all_events.items():
            for event in events:
                event_start = date_parser.parse(event['StartTime'])
                event_end = date_parser.parse(event['EndTime'])
                
                # Check overlap
                if slot_start < event_end and slot_end > event_start:
                    return True
        
        return False
    
    def _heuristic_fallback(self, attendees: List[str], duration_minutes: int, 
                           time_preference: str, start_date: str) -> Tuple[str, str]:
        """Advanced heuristic algorithm when OR-Tools fails."""
        print("🔍 Using advanced heuristic optimization...")
        
        # Heuristic 1: Greedy earliest-fit with preference weighting
        search_date = date_parser.parse(start_date).date()
        
        # Get all events
        all_events = {}
        for attendee in attendees:
            events = self.get_calendar_events(
                attendee, start_date, 
                (search_date + timedelta(days=7)).isoformat()
            )
            all_events[attendee] = events
        
        # Search strategy based on preference
        if time_preference == "morning":
            time_ranges = [(9, 12), (12, 17), (17, 18)]
        elif time_preference == "afternoon":
            time_ranges = [(12, 17), (9, 12), (17, 18)]
        elif time_preference == "evening":
            time_ranges = [(17, 18), (12, 17), (9, 12)]
        else:
            time_ranges = [(9, 12), (12, 17), (17, 18)]
        
        # Try each day
        for day_offset in range(7):
            current_date = search_date + timedelta(days=day_offset)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Try each time range in preference order
            for start_hour, end_hour in time_ranges:
                slot = self._find_day_slot(
                    all_events, current_date, duration_minutes, start_hour, end_hour
                )
                if slot:
                    print(f"✅ Heuristic found slot: {slot[0]}")
                    return slot
        
        # Ultimate fallback
        return self._get_emergency_slot(start_date, duration_minutes)
    
    def _find_optimal_slot_traditional(self, attendees: List[str], duration_minutes: int, 
                                      time_preference: str, start_date: str) -> Tuple[str, str]:
        """Traditional slot finding algorithm."""
        return _optimal(attendees, duration_minutes, start_date)
    
    def _get_emergency_slot(self, start_date: str, duration_minutes: int) -> Tuple[str, str]:
        """Emergency slot when all else fails."""
        start_dt = date_parser.parse(start_date)
        if start_dt.hour >= 18:
            # Schedule for next day morning
            start_dt = start_dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            # Schedule for later today
            start_dt = start_dt.replace(hour=max(start_dt.hour + 1, 9), minute=0, second=0, microsecond=0)
        
        start_dt = self.timezone.localize(start_dt.replace(tzinfo=None))
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        
        return start_dt.isoformat(), end_dt.isoformat()
    
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
        Main function to process a meeting request with Agentic AI enhancements
        
        Args:
            input_data: Input JSON with meeting request
            
        Returns:
            Output JSON with scheduled meeting details
        """
        try:
            print("🚀 Starting Agentic AI Meeting Processing...")
            print(f"📧 Processing request from: {input_data.get('From', 'Unknown')}")
            
            # Extract attendee emails
            attendee_emails = [input_data['From']]
            for attendee in input_data.get('Attendees', []):
                attendee_emails.append(attendee['email'])
            
            # Extract meeting information using Agentic AI
            email_content = input_data.get('EmailContent', input_data.get('Content', ''))
            subject = input_data.get('Subject', '')
            
            if self.agentic_mode and self.agentic_ready:
                print("🤖 Using Agentic AI for information extraction...")
                parsed_info = self._extract_meeting_info_with_agent(email_content)
                
                # Add context from input data
                parsed_info.update({
                    'subject': subject,
                    'from_email': input_data.get('From'),
                    'attendees': attendee_emails
                })
            else:
                print("📝 Using traditional LLM extraction...")
                parsed_info = self.parse_email_content(email_content, subject, attendee_emails)
            
            print(f"📊 Extracted meeting info: {parsed_info}")
            
            # Find optimal meeting slot using OR-based optimization
            print("🔍 Finding optimal meeting slot...")
            start_time, end_time = self.find_optimal_slot(
                attendee_emails,
                parsed_info.get('meeting_duration_minutes', 60),
                parsed_info.get('time_preference', 'anytime'),
                input_data.get('Datetime', datetime.now(self.timezone).isoformat())
            )
            
            print(f"⏰ Optimal slot found: {start_time} to {end_time}")
            
            # Generate final output
            output = self.generate_output_json(input_data, parsed_info, start_time, end_time)
            
            print("✅ Agentic scheduling completed successfully!")
            return output
            
        except Exception as e:
            print(f"❌ Error processing meeting request: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback
            return {
                "error": "Scheduling failed",
                "message": str(e),
                "fallback_suggestion": "Please try again or contact support"
            }
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
