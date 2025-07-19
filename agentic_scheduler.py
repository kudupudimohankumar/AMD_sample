# ------------------------------------------------------------------
#  NEW IMPORTS for the OR-Tools engine
# ------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from ortools.sat.python import cp_model
import datetime as _dt
from dateutil import parser as _dp
import asyncio
import traceback

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
        start = _dp.parse(ev["StartTime"])
        end = _dp.parse(ev["EndTime"])
        
        # Ensure timezone-aware datetimes
        if start.tzinfo is None:
            start = IST.localize(start)
        else:
            start = start.astimezone(IST)
            
        if end.tzinfo is None:
            end = IST.localize(end)
        else:
            end = end.astimezone(IST)
            
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
        Extract meeting information using Pydantic AI agents with MCP tools.
        
        Args:
            email_content: Raw email content
            
        Returns:
            Dict containing extracted meeting parameters
        """
        if not self.agentic_mode or not self.agentic_ready:
            return self._extract_meeting_info_traditional(email_content)
        
        try:
            from pydantic_ai import Agent, Tool
            from pydantic_ai.mcp import MCPServerStdio
            from pydantic import BaseModel
            from typing import Optional, List
            import asyncio
            from datetime import datetime, timedelta
            import re
            
            class MeetingInfo(BaseModel):
                meeting_duration_minutes: int
                time_preference: str  # morning, afternoon, evening, anytime
                urgency: str  # high, medium, low
                meeting_type: str  # team_meeting, client_call, interview, presentation, other
                preferred_date: Optional[str] = None
                attendees_count: Optional[int] = None
                location_preference: Optional[str] = None
                recurring: bool = False
                subject_keywords: Optional[List[str]] = None
            
            # Define MCP servers for enhanced extraction
            time_server = MCPServerStdio(
                "python",
                args=["-m", "mcp_server_time", "--local-timezone=Asia/Kolkata"],
            )
            
            # Define extraction tools
            @Tool
            def extract_duration_from_text(text: str) -> Dict[str, Any]:
                """Extract meeting duration from text using pattern matching."""
                duration_patterns = [
                    (r'(\d+)\s*hours?', lambda x: int(x) * 60),
                    (r'(\d+)\s*hrs?', lambda x: int(x) * 60),
                    (r'(\d+)\s*minutes?', lambda x: int(x)),
                    (r'(\d+)\s*mins?', lambda x: int(x)),
                    (r'half\s*hour', lambda x: 30),
                    (r'quarter\s*hour', lambda x: 15),
                    (r'1\.5\s*hours?', lambda x: 90),
                    (r'two\s*hours?', lambda x: 120),
                    (r'one\s*hour', lambda x: 60),
                ]
                
                for pattern, converter in duration_patterns:
                    match = re.search(pattern, text.lower())
                    if match:
                        if pattern in [r'half\s*hour', r'quarter\s*hour', r'1\.5\s*hours?', r'two\s*hours?', r'one\s*hour']:
                            return {"duration": converter(None), "confidence": "high"}
                        else:
                            return {"duration": converter(match.group(1)), "confidence": "high"}
                
                # Default duration based on meeting type keywords
                if any(word in text.lower() for word in ['standup', 'daily', 'scrum']):
                    return {"duration": 15, "confidence": "medium"}
                elif any(word in text.lower() for word in ['interview', 'presentation']):
                    return {"duration": 45, "confidence": "medium"}
                elif any(word in text.lower() for word in ['workshop', 'training']):
                    return {"duration": 120, "confidence": "medium"}
                
                return {"duration": 60, "confidence": "low"}  # Default
            
            @Tool
            def extract_time_preference(text: str) -> Dict[str, str]:
                """Extract time preferences from meeting text."""
                text_lower = text.lower()
                
                # Morning indicators
                if any(word in text_lower for word in ['morning', 'am', '9am', '10am', '11am', 'early']):
                    return {"preference": "morning", "confidence": "high"}
                
                # Afternoon indicators
                elif any(word in text_lower for word in ['afternoon', 'pm', '1pm', '2pm', '3pm', '4pm', 'lunch']):
                    return {"preference": "afternoon", "confidence": "high"}
                
                # Evening indicators
                elif any(word in text_lower for word in ['evening', '5pm', '6pm', 'eod', 'end of day']):
                    return {"preference": "evening", "confidence": "high"}
                
                # Specific day preferences
                days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                for day in days:
                    if day in text_lower:
                        return {"preference": day, "confidence": "high"}
                
                # Time flexibility indicators
                if any(word in text_lower for word in ['flexible', 'anytime', 'convenient', 'whenever']):
                    return {"preference": "anytime", "confidence": "medium"}
                
                return {"preference": "anytime", "confidence": "low"}
            
            @Tool
            def extract_urgency_level(text: str) -> Dict[str, str]:
                """Extract urgency level from meeting text."""
                text_lower = text.lower()
                
                # High urgency indicators
                if any(word in text_lower for word in ['urgent', 'asap', 'immediately', 'emergency', 'critical']):
                    return {"urgency": "high", "confidence": "high"}
                
                # Medium urgency indicators
                elif any(word in text_lower for word in ['soon', 'this week', 'priority', 'important']):
                    return {"urgency": "medium", "confidence": "high"}
                
                # Low urgency indicators
                elif any(word in text_lower for word in ['flexible', 'whenever', 'no rush', 'convenient']):
                    return {"urgency": "low", "confidence": "high"}
                
                return {"urgency": "medium", "confidence": "medium"}  # Default
            
            @Tool
            def extract_meeting_type(text: str) -> Dict[str, str]:
                """Extract meeting type from content."""
                text_lower = text.lower()
                
                # Team meeting indicators
                if any(word in text_lower for word in ['team', 'standup', 'scrum', 'sprint', 'retrospective']):
                    return {"type": "team_meeting", "confidence": "high"}
                
                # Client call indicators
                elif any(word in text_lower for word in ['client', 'customer', 'external', 'demo', 'sales']):
                    return {"type": "client_call", "confidence": "high"}
                
                # Interview indicators
                elif any(word in text_lower for word in ['interview', 'candidate', 'hiring', 'screening']):
                    return {"type": "interview", "confidence": "high"}
                
                # Presentation indicators
                elif any(word in text_lower for word in ['presentation', 'demo', 'showcase', 'review']):
                    return {"type": "presentation", "confidence": "high"}
                
                # Planning indicators
                elif any(word in text_lower for word in ['planning', 'strategy', 'roadmap', 'brainstorm']):
                    return {"type": "planning", "confidence": "high"}
                
                return {"type": "other", "confidence": "medium"}
            
            # Create the specialized extraction agent
            extraction_agent = Agent(
                model=self.agent_model,
                result_type=MeetingInfo,
                mcp_servers=[time_server],
                tools=[extract_duration_from_text, extract_time_preference, extract_urgency_level, extract_meeting_type],
                system_prompt="""You are an expert meeting information extraction agent with access to specialized tools.

Your task is to analyze email content and extract meeting parameters accurately.

Use the available tools to:
1. extract_duration_from_text: Get meeting duration from text
2. extract_time_preference: Determine preferred meeting time
3. extract_urgency_level: Assess meeting urgency
4. extract_meeting_type: Classify the meeting type
5. get_current_time: Get current date/time for context

Process the email content systematically:
- First, call get_current_time to understand the current context
- Then use each extraction tool to gather specific information
- Synthesize the results into a comprehensive MeetingInfo object
- Ensure all extracted values are realistic and consistent

Be thorough and precise in your analysis."""
            )
            
            # Run extraction with the agent
            async def run_extraction():
                async with extraction_agent.run_mcp_servers():
                    result = await extraction_agent.run(
                        f"Extract comprehensive meeting information from this email content:\n\n{email_content}\n\nUse all available tools to gather accurate details."
                    )
                    return result.data.model_dump()
            
            # Execute the extraction
            try:
                # Check if we're already in an async context
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                
                if loop is not None:
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, run_extraction())
                        extracted_info = future.result(timeout=10)
                else:
                    # Not in an async context, safe to use asyncio.run
                    extracted_info = asyncio.run(run_extraction())
                
                print(f"🤖 Agentic extraction complete: {extracted_info}")
                return extracted_info
                
            except Exception as async_error:
                print(f"⚠️ Async extraction failed: {async_error}")
                # Try sync fallback if available
                try:
                    print("🔄 Attempting sync extraction...")
                    sync_agent = Agent(
                        model=self.agent_model,
                        result_type=MeetingInfo,
                        tools=[extract_duration_from_text, extract_time_preference, extract_urgency_level, extract_meeting_type],
                        system_prompt="Extract meeting information from email content. Be concise and accurate."
                    )
                    
                    # Use run_sync for synchronous execution
                    result = sync_agent.run_sync(
                        f"Extract meeting information from this email:\n\n{email_content}"
                    )
                    
                    extracted_info = result.data.model_dump()
                    print(f"🤖 Sync agent extracted: {extracted_info}")
                    return extracted_info
                    
                except Exception as sync_error:
                    print(f"⚠️ Sync extraction also failed: {sync_error}")
                    return self._extract_meeting_info_traditional(email_content)
            
        except Exception as e:
            print(f"⚠️ Agentic extraction setup failed: {e}")
            return self._extract_meeting_info_traditional(email_content)
    
    def _extract_meeting_info_traditional(self, email_content: str) -> Dict[str, Any]:
        """Enhanced traditional extraction with comprehensive pattern matching."""
        
        def extract_duration(text: str) -> int:
            """Extract duration with advanced pattern matching."""
            duration_patterns = [
                (r'(\d+)\s*hours?', lambda x: int(x) * 60),
                (r'(\d+)\s*hrs?', lambda x: int(x) * 60),
                (r'(\d+)\s*minutes?', lambda x: int(x)),
                (r'(\d+)\s*mins?', lambda x: int(x)),
                (r'half\s*hour', lambda x: 30),
                (r'quarter\s*hour', lambda x: 15),
                (r'1\.5\s*hours?', lambda x: 90),
                (r'two\s*hours?', lambda x: 120),
                (r'one\s*hour', lambda x: 60),
                (r'thirty\s*minutes?', lambda x: 30),
                (r'fifteen\s*minutes?', lambda x: 15),
            ]
            
            for pattern, converter in duration_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    if pattern in [r'half\s*hour', r'quarter\s*hour', r'1\.5\s*hours?', 
                                   r'two\s*hours?', r'one\s*hour', r'thirty\s*minutes?', r'fifteen\s*minutes?']:
                        return converter(None)
                    else:
                        return converter(match.group(1))
            
            # Meeting type-based duration estimation
            text_lower = text.lower()
            if any(word in text_lower for word in ['standup', 'daily', 'scrum', 'check-in']):
                return 15
            elif any(word in text_lower for word in ['interview', 'screening']):
                return 45
            elif any(word in text_lower for word in ['workshop', 'training', 'session']):
                return 120
            elif any(word in text_lower for word in ['quick', 'brief', 'short']):
                return 30
            elif any(word in text_lower for word in ['long', 'detailed', 'comprehensive']):
                return 90
            
            return 60  # Default
        
        def extract_time_preference(text: str) -> str:
            """Extract time preference with comprehensive patterns."""
            text_lower = text.lower()
            
            # Specific time mentions
            if re.search(r'\b(9|10|11)\s*(am|a\.m\.)', text_lower):
                return "morning"
            elif re.search(r'\b(12|1|2|3|4)\s*(pm|p\.m\.)', text_lower):
                return "afternoon"
            elif re.search(r'\b(5|6|7)\s*(pm|p\.m\.)', text_lower):
                return "evening"
            
            # General time preferences
            morning_words = ['morning', 'early', 'start of day', 'beginning', 'am']
            afternoon_words = ['afternoon', 'midday', 'lunch time', 'post lunch', 'pm']
            evening_words = ['evening', 'end of day', 'eod', 'late', 'after hours']
            flexible_words = ['flexible', 'anytime', 'convenient', 'whenever', 'open']
            
            if any(word in text_lower for word in morning_words):
                return "morning"
            elif any(word in text_lower for word in afternoon_words):
                return "afternoon"
            elif any(word in text_lower for word in evening_words):
                return "evening"
            elif any(word in text_lower for word in flexible_words):
                return "anytime"
            
            # Day-specific preferences
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for day in days:
                if day in text_lower:
                    return day
            
            return "anytime"
        
        def extract_urgency(text: str) -> str:
            """Extract urgency level with comprehensive patterns."""
            text_lower = text.lower()
            
            high_urgency = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 
                           'priority', 'rush', 'today', 'now', 'must', 'need to discuss']
            low_urgency = ['flexible', 'whenever', 'no rush', 'convenient', 'eventually',
                          'when available', 'leisure', 'no hurry', 'relaxed', 'casual']
            medium_urgency = ['soon', 'this week', 'important', 'needed', 'required',
                             'timely', 'prompt', 'next week']
            
            # Check for explicit low urgency first (more specific)
            if any(word in text_lower for word in low_urgency):
                return "low"
            elif any(word in text_lower for word in high_urgency):
                return "high"
            elif any(word in text_lower for word in medium_urgency):
                return "medium"
            
            return "medium"  # Default
        
        def extract_meeting_type(text: str) -> str:
            """Extract meeting type with comprehensive patterns."""
            text_lower = text.lower()
            
            # Team meeting indicators
            team_words = ['team', 'standup', 'scrum', 'sprint', 'retrospective', 
                         'all hands', 'staff', 'department']
            
            # Client/external indicators  
            client_words = ['client', 'customer', 'external', 'demo', 'sales',
                           'prospect', 'vendor', 'partner']
            
            # Interview indicators
            interview_words = ['interview', 'candidate', 'hiring', 'screening',
                              'recruitment', 'onboarding']
            
            # Presentation indicators
            presentation_words = ['presentation', 'demo', 'showcase', 'review',
                                 'pitch', 'proposal', 'walkthrough']
            
            # Planning indicators
            planning_words = ['planning', 'strategy', 'roadmap', 'brainstorm',
                             'ideation', 'design', 'architecture']
            
            if any(word in text_lower for word in team_words):
                return "team_meeting"
            elif any(word in text_lower for word in client_words):
                return "client_call"
            elif any(word in text_lower for word in interview_words):
                return "interview"
            elif any(word in text_lower for word in presentation_words):
                return "presentation"
            elif any(word in text_lower for word in planning_words):
                return "planning"
            
            return "other"
        
        # Extract information using enhanced pattern matching
        extracted_info = {
            "meeting_duration_minutes": extract_duration(email_content),
            "time_preference": extract_time_preference(email_content),
            "urgency": extract_urgency(email_content),
            "meeting_type": extract_meeting_type(email_content)
        }
        
        # Try LLM extraction if available
        try:
            prompt = f"""Extract meeting information from this email and return ONLY a JSON object:

Email: {email_content}

Return JSON with exactly these fields:
{{"meeting_duration_minutes": {extracted_info['meeting_duration_minutes']}, "time_preference": "{extracted_info['time_preference']}", "urgency": "{extracted_info['urgency']}", "meeting_type": "{extracted_info['meeting_type']}"}}

Improve the values if the email contains more specific information. Return only the JSON:"""

            response = requests.post(f"{self.vllm_url}/completions", 
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.1
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['text'].strip()
                import json
                try:
                    llm_info = json.loads(result)
                    # Merge LLM results with pattern-based extraction
                    extracted_info.update(llm_info)
                except Exception as e:
                    print(f"LLM JSON parsing failed: {e}")
                    
        except Exception as e:
            print(f"LLM extraction failed: {e}")
        
        print(f"📊 Traditional extraction result: {extracted_info}")
        return extracted_info
    
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
            # Clean and validate date format
            def clean_date(date_str: str) -> str:
                """Ensure date is in YYYY-MM-DD format"""
                try:
                    # If it's already a datetime string, extract just the date part
                    if 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    
                    # Parse and reformat to ensure correct format
                    parsed_date = date_parser.parse(date_str).date()
                    return parsed_date.isoformat()  # Returns YYYY-MM-DD
                except Exception as e:
                    print(f"Date parsing error for {date_str}: {e}")
                    # Fallback to current date
                    return datetime.now().date().isoformat()
            
            clean_start = clean_date(start_date)
            clean_end = clean_date(end_date)
            
            # Convert dates to proper format for Google Calendar API
            start_datetime = f"{clean_start}T00:00:00+05:30"  # IST timezone
            end_datetime = f"{clean_end}T23:59:59+05:30"     # IST timezone
            
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
            BUSINESS_START_HOUR = 9  # 9 AM
            BUSINESS_END_HOUR = 18   # 6 PM
            
            # Calculate slots only for business hours to avoid hour > 23 errors
            BUSINESS_HOURS = BUSINESS_END_HOUR - BUSINESS_START_HOUR  # 9 hours
            SLOTS_PER_DAY = (BUSINESS_HOURS * 60) // SLOT_DURATION  # 36 slots (9 hours * 4 slots/hour)
            DAYS_TO_SEARCH = 7
            
            print(f"🔧 OR-Tools setup: {SLOTS_PER_DAY} slots/day, {DAYS_TO_SEARCH} days")
            
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
                        all_events, current_date, slot, slots_needed, SLOT_DURATION, BUSINESS_START_HOUR
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
                            # Convert back to datetime with business hours offset
                            selected_date = search_date + timedelta(days=day)
                            
                            # Calculate actual time considering business hours start
                            slot_minutes_from_business_start = slot * SLOT_DURATION
                            actual_hour = BUSINESS_START_HOUR + (slot_minutes_from_business_start // 60)
                            actual_minute = slot_minutes_from_business_start % 60
                            
                            # Validate hour is within business hours and valid range
                            if actual_hour >= BUSINESS_END_HOUR or actual_hour >= 24:
                                print(f"⚠️ Invalid hour calculated: {actual_hour}, using fallback")
                                return self._heuristic_fallback(attendees, duration_minutes, time_preference, start_date)
                            
                            # Ensure minute is within valid range
                            actual_minute = min(actual_minute, 59)
                            
                            try:
                                start_time = self.timezone.localize(
                                    datetime.combine(selected_date, time(actual_hour, actual_minute))
                                )
                                end_time = start_time + timedelta(minutes=duration_minutes)
                                
                                print(f"✅ OR-Tools found optimal slot: {start_time}")
                                return start_time.isoformat(), end_time.isoformat()
                            except ValueError as ve:
                                print(f"⚠️ Time creation error: {ve}, using fallback")
                                return self._heuristic_fallback(attendees, duration_minutes, time_preference, start_date)
            
            print("⚠️ OR-Tools couldn't find optimal solution, using heuristic fallback")
            return self._heuristic_fallback(attendees, duration_minutes, time_preference, start_date)
            
        except Exception as e:
            print(f"❌ OR optimization failed: {e}")
            return self._find_optimal_slot_traditional(attendees, duration_minutes, time_preference, start_date)
    
    def _get_time_preference_weights(self, time_preference: str) -> Dict[int, int]:
        """Get optimization weights based on time preference (business hours 9 AM - 6 PM)."""
        weights = {}
        SLOT_DURATION = 15
        BUSINESS_START_HOUR = 9
        
        if time_preference == "morning":
            # Weight morning slots (9 AM - 12 PM) higher - slots 0-11
            for hour in range(9, 12):
                for min_slot in range(0, 60, SLOT_DURATION):
                    # Convert to business hours slot (offset by 9 AM)
                    slot = (((hour - BUSINESS_START_HOUR) * 60) + min_slot) // SLOT_DURATION
                    weights[slot] = 10
        elif time_preference == "afternoon":
            # Weight afternoon slots (12 PM - 5 PM) higher - slots 12-31
            for hour in range(12, 17):
                for min_slot in range(0, 60, SLOT_DURATION):
                    # Convert to business hours slot (offset by 9 AM)
                    slot = (((hour - BUSINESS_START_HOUR) * 60) + min_slot) // SLOT_DURATION
                    weights[slot] = 10
        elif time_preference == "evening":
            # Weight late afternoon/early evening (4 PM - 6 PM) higher - slots 28-35
            for hour in range(16, 18):
                for min_slot in range(0, 60, SLOT_DURATION):
                    # Convert to business hours slot (offset by 9 AM)
                    slot = (((hour - BUSINESS_START_HOUR) * 60) + min_slot) // SLOT_DURATION
                    weights[slot] = 10
        
        return weights
    
    def _check_slot_conflicts(self, all_events: Dict, date: datetime, 
                             start_slot: int, slots_needed: int, slot_duration: int, 
                             business_start_hour: int = 9) -> bool:
        """Check if a time slot conflicts with existing meetings."""
        
        # Calculate actual time considering business hours offset
        slot_minutes_from_business_start = start_slot * slot_duration
        start_hour = business_start_hour + (slot_minutes_from_business_start // 60)
        start_min = slot_minutes_from_business_start % 60
        
        end_slot_minutes = (start_slot + slots_needed) * slot_duration
        end_hour = business_start_hour + (end_slot_minutes // 60)
        end_min = end_slot_minutes % 60
        
        # Skip if hours are invalid (beyond business hours or 24-hour format)
        if start_hour >= 24 or end_hour >= 24 or start_hour >= 18 or end_hour >= 18:
            return True  # Treat as conflict to avoid the slot
        
        # Ensure minutes are within valid range
        start_min = min(start_min, 59)
        end_min = min(end_min, 59)
        
        try:
            slot_start = self.timezone.localize(
                datetime.combine(date, time(start_hour, start_min))
            )
            slot_end = self.timezone.localize(
                datetime.combine(date, time(end_hour, end_min))
            )
        except ValueError:
            # If time creation fails, treat as conflict
            return True
        
        for attendee, events in all_events.items():
            for event in events:
                event_start = date_parser.parse(event['StartTime'])
                event_end = date_parser.parse(event['EndTime'])
                
                # Ensure both datetimes are timezone-aware for comparison
                if event_start.tzinfo is None:
                    event_start = self.timezone.localize(event_start)
                else:
                    event_start = event_start.astimezone(self.timezone)
                
                if event_end.tzinfo is None:
                    event_end = self.timezone.localize(event_end)
                else:
                    event_end = event_end.astimezone(self.timezone)
                
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
                
                # Ensure timezone consistency for comparison
                if event_start.tzinfo is None:
                    event_start = self.timezone.localize(event_start)
                else:
                    event_start = event_start.astimezone(self.timezone)
                
                if event_end.tzinfo is None:
                    event_end = self.timezone.localize(event_end)
                else:
                    event_end = event_end.astimezone(self.timezone)
                
                # Ensure slot times are timezone-aware
                if slot_start.tzinfo is None:
                    slot_start = self.timezone.localize(slot_start)
                if slot_end.tzinfo is None:
                    slot_end = self.timezone.localize(slot_end)
                
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
            
            # Get start date - ensure proper format (YYYY-MM-DD)
            start_date = input_data.get('Datetime', datetime.now(self.timezone).date().isoformat())
            if 'T' in start_date:  # Handle full datetime strings
                start_date = start_date.split('T')[0]
            
            start_time, end_time = self.find_optimal_slot(
                attendee_emails,
                parsed_info.get('meeting_duration_minutes', 60),
                parsed_info.get('time_preference', 'anytime'),
                start_date
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
