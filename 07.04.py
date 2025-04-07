import logging
import asyncio
import httpx
import mimetypes
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.filters import Command
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os
load_dotenv()
TELEGRAM_BOT_TOKEN=os.getenv('TELEGRAM_API_TOKEN')
OKDESK_API_TOKEN=os.getenv('OKDESK_API_TOKEN')
OKDESK_DOMAIN=os.getenv('OKDESK_DOMAIN')
# Constants - Direct values instead of environment variables

OKDESK_API_URL = f"https://{OKDESK_DOMAIN}/api/v1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("__main__")

# Global dictionary to store comment monitoring tasks
comment_tasks = {}
# Define states for conversation
class UserStates(StatesGroup):
    waiting_for_contact = State()
    waiting_for_issue = State()
    waiting_for_confirmation = State()
    waiting_for_follow_up = State()
    waiting_for_comment = State()
    selecting_issue = State()
    writing_comment = State()

async def find_contact_issues(
    contact_id: int,
    with_closed: bool = False,
    api_token: Optional[str] = None
) -> Union[List[Dict], None]:
    """
    Fetches issues associated with a specific contact ID in Okdesk.
    
    Args:
        contact_id (int): The ID of the contact whose issues are being retrieved.
        with_closed (bool, optional): Whether to include closed issues. Defaults to False.
        api_token (str, optional): API token. Defaults to OKDESK_API_TOKEN.
    
    Returns:
        List[Dict] or None: List of issues associated with the contact, or None if no issues are found.
    """
    if api_token is None:
        api_token = OKDESK_API_TOKEN
    
    issues_url = "https://emanengineering.okdesk.ru/api/v1/issues/list"
    
    params = {
        "api_token": api_token,
        "contact_ids[]": contact_id,  # API ожидает массив, поэтому `contact_ids[]`
        "with_closed": str(with_closed).lower()
    }
    
    logger.info(f"Fetching issues for Contact ID: {contact_id}")
    logger.debug(f"Request Params: {params}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(issues_url, params=params)
            
            logger.debug(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                issues = response.json()
                logger.info(f"Found {len(issues)} issues for contact {contact_id}")
                return issues
            elif response.status_code == 414:
                logger.error("Request URI too long. Check if contact_id is properly formatted.")
            else:
                logger.warning(f"Unexpected response: {response.status_code} - {response.text}")
            
    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as e:
        logger.error(f"Error fetching contact issues: {e}")
    
    return None


async def validate_issue_access(issue_id: int, contact_id: int) -> bool:
    """
    Validate if the contact has access to the specific issue
    
    Args:
        issue_id (int): The ID of the issue to check
        contact_id (int): The contact ID attempting to access the issue
    
    Returns:
        bool: True if the contact can access the issue, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch issue details to verify ownership
            params = {
                "api_token": OKDESK_API_TOKEN
            }
                
            response = await client.get(
                f"{OKDESK_API_URL}/issues/{issue_id}", 
                params=params
            )
            
            if response.status_code == 200:
                issue_data = response.json()
                # Check if the contact ID matches the issue's contact ID
                return str(issue_data.get('contact', {}).get('id')) == str(contact_id)
            
            return False
    
    except Exception as e:
        logger.error(f"Error validating issue access: {e}")
        return False

import logging
import asyncio
import httpx
import mimetypes
import io
from datetime import datetime
from typing import Optional

# Assuming validate_issue_access is defined elsewhere

async def add_comment_to_issue(
    issue_id: int,
    comment_text: str,
    contact_id: int,
    photo_data: Optional[bytes] = None,
    photo_filename: Optional[str] = None
) -> dict | None:
    """
    Add a comment to an existing issue in Okdesk with an optional photo attachment.

    Args:
        issue_id: The ID of the issue to comment on.
        comment_text: The text content of the comment.
        contact_id: The ID of the contact making the comment.
        photo_data: Binary data of the photo (if any).
        photo_filename: The original filename of the photo (if any).

    Returns:
        dict: The API response if successful, None otherwise.
    """
    # Validate access
    if not await validate_issue_access(issue_id, contact_id):
        logger.warning(f"Unauthorized comment attempt. Contact {contact_id} tried to comment on issue {issue_id}")
        return None

    logger.info(f"Sending comment to issue {issue_id} by contact {contact_id}")

    # Prepare the comment data
    data = {
        "comment[content]": comment_text,
        "comment[public]": "true",
        "comment[author_id]": str(contact_id),
        "comment[author_type]": "contact"
    }

    # Prepare files for multipart/form-data (if a photo is provided)
    files = {}
    if photo_data and photo_filename:
        # Guess the content type based on the filename
        content_type, _ = mimetypes.guess_type(photo_filename)
        if not content_type:
            content_type = "application/octet-stream"

        # Create a file object from the photo data
        file_obj = io.BytesIO(photo_data)
        unique_filename = f"{int(datetime.now().timestamp())}_{photo_filename}"

        # Structure the attachment as an object within an array
        files = {
            "comment[attachments][0][attachment]": (unique_filename, file_obj, content_type)
        }
        # Optional: Add metadata like description if required
        # data["comment[attachments][0][description]"] = "Photo from Telegram"
        logger.info(f"Attaching photo: {unique_filename} ({content_type})")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OKDESK_API_URL}/issues/{issue_id}/comments",
                params={"api_token": OKDESK_API_TOKEN},
                data=data,
                files=files if files else None  # Only include files if there’s an attachment
            )
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            result = response.json()
            logger.info(f"Comment creation response: {result}")
            return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error adding comment to issue {issue_id}: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as ex:
        logger.exception(f"Unexpected error adding comment to issue {issue_id}: {ex}")
        return None






async def create_issues_keyboard(issues: List[Dict]) -> ReplyKeyboardMarkup:
    """
    Create a keyboard with issue titles for user selection.
    
    Args:
        issues (List[Dict]): List of issues to display
    
    Returns:
        ReplyKeyboardMarkup: Keyboard with issue titles
    """
    keyboard_buttons = []
    for issue in issues:
        # Format: "[Issue ID] Short Description"
        button_text = f"[{issue['id']}] {issue.get('title', 'Untitled Issue')[:30]}"
        keyboard_buttons.append([KeyboardButton(text=button_text)])
    
    keyboard_buttons.append([KeyboardButton(text="Cancel")])
    
    return ReplyKeyboardMarkup(
        keyboard=keyboard_buttons, 
        resize_keyboard=True, 
        one_time_keyboard=True
    )

# OKDesk API functions
async def find_contact_by_phone(phone_number: str) -> dict|None:
    """Search for a contact in OKDesk by phone number."""
    # Clean phone number (remove '+' and any spaces)
    phone = phone_number.lstrip('+').replace(' ', '')
    
    logger.info(f"Searching for contact with phone: {phone_number} (query: {phone})")
    
    params = {"api_token": OKDESK_API_TOKEN, "phone": phone}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{OKDESK_API_URL}/contacts/", params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Contact search response: {data}")
            
            if isinstance(data, list) and data:
                return data[0]
            elif isinstance(data, dict) and 'id' in data:
                return data
            return None
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error when searching for contact: {e}")
        if e.response:
            logger.error(f"Response status: {e.response.status_code}, body: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error when searching for contact: {e}")
        return None

async def create_contact(first_name: str, last_name: str, phone_number: str) -> dict|None:
    """Create a new contact in OKDesk."""
    # Clean phone number (remove '+' and any spaces)
    phone = phone_number.lstrip('+').replace(' ', '')
    
    logger.info(f"Creating new contact with phone: {phone_number} (submit: {phone})")
    
    # Prepare contact data
    contact_data = {
        "first_name": first_name,
        "last_name": last_name,
        "mobile_phone": phone,
    }
    
    params = {"api_token": OKDESK_API_TOKEN}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OKDESK_API_URL}/contacts/", 
                params=params, 
                json=contact_data
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Contact creation response: {data}")
            
            if "id" in data:
                return data
            return None
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error when creating contact: {e}")
        if e.response:
            logger.error(f"Response status: {e.response.status_code}, body: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error when creating contact: {e}")
        return None

async def create_issue(title: str, description: str, contact_id: int) -> dict | None:
    """Create a new issue in OKDesk."""
    # Using the correct field name 'title' instead of 'subject'
    issue_data = {
        "title": title,
        "description": description,
        "contact_id": contact_id
    }
    
    params = {"api_token": OKDESK_API_TOKEN}
    
    logger.info(f"Creating issue for contact ID {contact_id} with title: {title}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OKDESK_API_URL}/issues/", 
                params=params, 
                json=issue_data
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Issue creation response: {data}")
            
            if "id" in data:
                return data
            return None
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error when creating issue: {e}")
        if e.response:
            logger.error(f"Response status: {e.response.status_code}, body: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error when creating issue: {e}")
        return None

@dataclass
class CommentMonitoringTask:
    """
    A class to manage comment monitoring for a specific user and issue
    """
    user_id: int
    contact_id: int
    issue_id: int
    task: asyncio.Task
    created_at: datetime = field(default_factory=datetime.now)
    last_comment_ids: Dict[int, int] = field(default_factory=dict)
    
    def is_expired(self, max_duration: timedelta = timedelta(hours=24)) -> bool:
        """
        Check if the monitoring task has exceeded the maximum duration
        
        Args:
            max_duration (timedelta): Maximum time to monitor comments
        
        Returns:
            bool: True if the task is expired, False otherwise
        """
        return datetime.now() - self.created_at > max_duration

class CommentMonitorManager:
    """
    A comprehensive manager for handling comment monitoring tasks
    """
    def __init__(self, bot, dp):
        """
        Initialize the Comment Monitor Manager
        
        Args:
            bot (Bot): Telegram bot instance
            dp (Dispatcher): Aiogram dispatcher
        """
        self.tasks: Dict[str, CommentMonitoringTask] = {}
        self.bot = bot
        self.dp = dp
        self.logger = logging.getLogger(__name__)
        
        # Start a background task to clean up expired monitoring tasks
        self.cleanup_task = asyncio.create_task(self.periodic_task_cleanup())
    
    def generate_task_key(self, user_id: int, issue_id: int) -> str:
        """
        Generate a unique key for a monitoring task
        
        Args:
            user_id (int): Telegram user ID
            issue_id (int): Issue ID to monitor
        
        Returns:
            str: Unique task key
        """
        return f"{user_id}_{issue_id}"
    
    async def start_monitoring(
        self, 
        user_id: int, 
        contact_id: int, 
        issue_id: int
    ) -> CommentMonitoringTask:
        """
        Start monitoring comments for a specific issue
        
        Args:
            user_id (int): Telegram user ID
            contact_id (int): Contact ID in OKDesk
            issue_id (int): Issue ID to monitor
        
        Returns:
            CommentMonitoringTask: Created monitoring task
        """
        task_key = self.generate_task_key(user_id, issue_id)
        
        # Cancel existing task if it exists
        if task_key in self.tasks:
            await self.stop_monitoring(user_id, issue_id)
        
        # Create a new monitoring task
        async def comment_monitoring_loop():
            last_comment_ids = {}
            try:
                while True:
                    try:
                        last_comment_ids = await comment_notification_job_for_all_issues(
                            self.bot, user_id, contact_id, last_comment_ids
                        )
                        await asyncio.sleep(60)  # Check every 30 seconds
                    
                    except asyncio.CancelledError:
                        self.logger.info(f"Comment monitoring task for {task_key} was cancelled")
                        break
                    except Exception as e:
                        self.logger.error(f"Error in comment monitoring for {task_key}: {e}")
                        await asyncio.sleep(60)  # Wait longer on error
            
            except Exception as e:
                self.logger.error(f"Unexpected error in comment monitoring: {e}")
        
        task = asyncio.create_task(comment_monitoring_loop())
        
        # Create and store the monitoring task
        monitoring_task = CommentMonitoringTask(
            user_id=user_id, 
            contact_id=contact_id, 
            issue_id=issue_id, 
            task=task
        )
        
        self.tasks[task_key] = monitoring_task
        
        return monitoring_task
    
    async def stop_monitoring(self, user_id: int, issue_id: int) -> bool:
        """
        Stop monitoring comments for a specific issue
        
        Args:
            user_id (int): Telegram user ID
            issue_id (int): Issue ID to stop monitoring
        
        Returns:
            bool: True if task was found and cancelled, False otherwise
        """
        task_key = self.generate_task_key(user_id, issue_id)
        
        if task_key in self.tasks:
            monitoring_task = self.tasks[task_key]
            
            # Cancel the task
            if not monitoring_task.task.done():
                monitoring_task.task.cancel()
            
            # Remove from tasks dictionary
            del self.tasks[task_key]
            
            self.logger.info(f"Stopped monitoring for task {task_key}")
            return True
        
        return False
    
    async def periodic_task_cleanup(self):
        """
        Periodically clean up expired or completed monitoring tasks
        """
        while True:
            try:
                # Find expired or completed tasks
                expired_tasks = [
                    key for key, task in self.tasks.items()
                    if task.is_expired() or task.task.done()
                ]
                
                # Remove expired tasks
                for key in expired_tasks:
                    monitoring_task = self.tasks[key]
                    if not monitoring_task.task.done():
                        monitoring_task.task.cancel()
                    del self.tasks[key]
                
                # Log cleanup information
                if expired_tasks:
                    self.logger.info(f"Cleaned up {len(expired_tasks)} expired monitoring tasks")
                
                # Wait before next cleanup
                await asyncio.sleep(900)  # Check every 5 minutes
            
            except Exception as e:
                self.logger.error(f"Error in periodic task cleanup: {e}")
                await asyncio.sleep(900)
    
    async def shutdown(self):
        """
        Gracefully shut down all monitoring tasks
        """
        # Cancel the cleanup task
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
        
        # Cancel all monitoring tasks
        for task_key, monitoring_task in list(self.tasks.items()):
            if not monitoring_task.task.done():
                monitoring_task.task.cancel()
        
        # Clear the tasks dictionary
        self.tasks.clear()




















async def get_issue_comments(issue_id: int, since_time=None) -> list | None:
    """Get comments for an issue from OKDesk."""
    params = {"api_token": OKDESK_API_TOKEN}
    
    if since_time:
        params["created_since"] = since_time.isoformat()
    
    logger.info(f"Fetching comments for issue ID {issue_id}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OKDESK_API_URL}/issues/{issue_id}/comments", 
                params=params
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Retrieved {len(data) if isinstance(data, list) else 0} comments")
            
            # Filter for public comments only
            if isinstance(data, list):
                return [comment for comment in data if comment.get('public', False)]
            return []
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error when fetching comments: {e}")
        if e.response:
            logger.error(f"Response status: {e.response.status_code}, body: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Error when fetching comments: {e}")
        return None

async def comment_notification_job_for_all_issues(
    bot: Bot, 
    user_id: int, 
    contact_id: int, 
    last_comment_ids: Optional[Dict[int, int]] = None
) -> Dict[int, int]:
    """
    Job to check for new public comments on all open issues for a contact and forward them to the user.
    
    Args:
        bot (Bot): Telegram bot instance
        user_id (int): Telegram user ID to send notifications to
        contact_id (int): Contact ID to fetch issues for
        last_comment_ids (Dict[int, int], optional): Dict of last known comment IDs per issue
    
    Returns:
        Dict[int, int]: Updated dictionary of last known comment IDs per issue
    """
    # Initialize last_comment_ids if not provided
    if last_comment_ids is None:
        last_comment_ids = {}
    
    # Fetch all open issues for the contact
    issues = await find_contact_issues(contact_id, with_closed=False)
    
    if not issues:
        logger.info("No open issues found for the contact.")
        return last_comment_ids
    
    # Process comments for each issue
    for issue in issues:
        issue_id = issue.get('id')
        if not issue_id:
            continue
        
        # Get last known comment ID for this issue, default to None
        last_known_comment_id = last_comment_ids.get(issue_id, 0)
        
        # Fetch comments for the issue
        comments = await get_issue_comments(issue_id)
        
        if not comments or not isinstance(comments, list):
            continue
        
        # Sort comments by ID to ensure we process in order
        comments = sorted(comments, key=lambda x: x.get('id', 0))
        
        # Find comments after the last known comment
        new_comments = [
            comment for comment in comments 
            if comment.get('id', 0) > last_known_comment_id 
            and comment.get('source') != 'client_portal'  # Exclude comments from client portal
            and str(comment.get('author', {}).get('id', "")) != str(contact_id)  # Exclude comments from the contact
        ]
        
        # Send notifications for new comments
        for comment in new_comments:
            try:
                author = comment.get('author', {}).get('name', 'Support Agent')
                content = comment.get('content', 'No content')
                issue_url = f"https://{OKDESK_DOMAIN}/issues/{issue_id}"
                
                message = (
                    f"💬 <b><a href='{issue_url}'>{issue_id}</a> -buyurtmaga izoh qo'shildi {author} tomonidan</b>\n"
                    f"<i></i>\n\n"
                    f"{content}"
                )
                
                await bot.send_message(user_id, message, parse_mode="HTML")
                logger.info(f"Sent comment notification to user {user_id} for issue {issue_id}")
                
                # Update last known comment ID for this issue
                last_comment_ids[issue_id] = comment.get('id', last_known_comment_id)
            
            except Exception as e:
                logger.error(f"Failed to send comment notification for issue {issue_id}: {e}")
    
    return last_comment_ids

# Setup bot and dispatcher
async def main():
    """Main function to start the bot"""
    logger.info("Starting bot...")
    
    # Initialize bot and dispatcher with storage
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    
    # Create the comment monitoring manager
    comment_monitor = CommentMonitorManager(bot, dp)
    
    # Command handlers
    @dp.message(Command("start"))
    async def cmd_start(message: types.Message, state: FSMContext):
        """Handle /start command"""
        contact_button = KeyboardButton(text="📱 Kontaktni yuborish", request_contact=True)
        markup = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[contact_button]])
        
        await message.answer(
            "👋 Qo'llab-quvvatlash xizmatiga xush kelibsiz!\n\n"
            "Buyurtmani ro'yxatdan o'tkazish uchun kontakt ma'lumotingizni ulashing.",
            reply_markup=markup
        )
        await state.set_state(UserStates.waiting_for_contact)
    
    @dp.message(UserStates.waiting_for_contact, F.contact)
    async def handle_contact(message: types.Message, state: FSMContext):
        """Handle when a user shares their contact information"""
        contact = message.contact
        logger.info(f"Received contact: phone={contact.phone_number}, name={contact.first_name}, username={message.from_user.username}")
        
        # Search for existing contact
        contact_data = await find_contact_by_phone(contact.phone_number)
        
        if contact_data and 'id' in contact_data:
            # We found an existing contact
            contact_id = contact_data['id']
            logger.info(f"Using existing contact with ID: {contact_id}")
        else:
            # No contact found, create a new one
            logger.info(f"No existing contact found, creating new one for phone: {contact.phone_number}")
            last_name = message.from_user.last_name or "-"
            new_contact = await create_contact(
                first_name=contact.first_name or "-",
                last_name=last_name,
                phone_number=contact.phone_number
            )
            if not new_contact or 'id' not in new_contact:
                await message.answer("❌ Kontakt yaratishda xatolik yuz berdi. Iltimos, keyinroq qayta urinib ko'ring.")
                await state.clear()
                return
            
            contact_id = new_contact['id']
            logger.info(f"Created new contact with ID: {contact_id}")
        
        # Save the contact_id to state for future use
        await state.update_data(contact_id=contact_id, phone=contact.phone_number, name=contact.first_name)
        
        # Proceed with the conversation
        await message.answer(
            "✅ Kontakt topildi.\n\nMuammoingiz yoki so'rovingizni tasvirlab bering:",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.set_state(UserStates.waiting_for_issue)
    
    @dp.message(UserStates.waiting_for_issue)
    async def handle_issue_description(message: types.Message, state: FSMContext):
        """Handle the issue description from the user"""
        user_data = await state.get_data()
        contact_id = user_data.get('contact_id')
        user_name = user_data.get('name', 'User')
        contact_button = KeyboardButton(text="📱 Kontaktni yuborish", request_contact=True)
        markup = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[contact_button]])
        
        if not contact_id:
            logger.error("No contact_id found in state")
            await message.answer("❌ Avval kontakt ma'lumotlarini yuborish kerak.",
                                 reply_markup=markup)

            await state.set_state(UserStates.waiting_for_contact)
            return
        
        issue_description = message.text
        issue_title = f"{issue_description}"
        
        # Save issue description to state
        await state.update_data(issue_description=issue_description, issue_title=issue_title)
        
        # Ask for confirmation
        confirm_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="✅ Arizani yuborish")],
                [KeyboardButton(text="❌ Bekor qilish")]
            ],
            resize_keyboard=True
        )
        
        confirmation_text = (
            f"Arizani yuborishni tasdiqlang:\n\n"
            f"Tavsif: {issue_description}\n\n"
            f"Arizani yuborish?"
        )
        
        await message.answer(confirmation_text, reply_markup=confirm_keyboard)
        await state.set_state(UserStates.waiting_for_confirmation)
    
        # Modify the handle_confirmation method to use the new manager
    @dp.message(UserStates.waiting_for_confirmation, F.text.in_(["✅ Arizani yuborish", "❌ Bekor qilish"]))
    async def handle_confirmation(message: types.Message, state: FSMContext, bot: Bot):
        """Handle user confirmation of the issue"""
        if message.text == "❌ Bekor qilish":
            # Show options for follow-up actions
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="💭 Izoh qo'shish")],
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            await message.answer(
                "Ariza bekor qilindi. Siz yangi ariza boshlashingiz mumkin.",
                reply_markup=follow_up_keyboard
            )
            await state.set_state(UserStates.waiting_for_follow_up)
            return
        
        # User confirmed, create the issue
        user_data = await state.get_data()
        contact_id = user_data.get('contact_id')
        description = user_data.get('issue_description')
        title = user_data.get('issue_title')
        
        # Create the issue in OKDesk
        issue_result = await create_issue(title, description, contact_id)
        
        if issue_result and 'id' in issue_result:
            issue_id = issue_result['id']
            issue_url = f"https://{OKDESK_DOMAIN}/issues/{issue_id}"
            
            # Save issue_id to state
            await state.update_data(last_issue_id=issue_id)
            
            # Create follow-up keyboard
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="💭 Izoh qo'shish")],
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            
            await message.answer(
                f"✅ Sizning arizangiz muvaffaqiyatli yaratildi!\n\n"
                f"Ariza raqami: {issue_id}\n"
                f"Arizaga havola: {issue_url} \n\n"
                f"Tez orada siz bilan bog'lanamiz.",
                reply_markup=follow_up_keyboard
            )
            
            # Start comment monitoring task for this issue
            user_id = message.from_user.id
            
            # Create a new async task for comment monitoring
            async def comment_check_task():
                last_comment_ids = {}
                try:
                    while True:
                        try:
                            last_comment_ids = await comment_notification_job_for_all_issues(
                                bot, user_id, contact_id, last_comment_ids
                            )
                            await asyncio.sleep(30)
                        except asyncio.CancelledError:
                            logger.info(f"Comment checking task cancelled for user {user_id}")
                            break
                        except Exception as e:
                            logger.error(f"Error in comment notification loop for user {user_id}: {e}")
                            await asyncio.sleep(60)
                except Exception as unexpected_error:
                    logger.error(f"Unexpected error in comment monitoring: {unexpected_error}")
            
            # Create the task
            comment_monitoring_task = asyncio.create_task(comment_check_task())
            comment_tasks[user_id] = comment_monitoring_task
        else:
            # Show options for follow-up actions even if creation failed
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            await message.answer(
                "❌ Buyurtma yaratishda xatolik yuz berdi. Iltimos, qayta urinib ko'ring.",
                reply_markup=follow_up_keyboard
            )
        
        await state.set_state(UserStates.waiting_for_follow_up)

    def get_comment_monitor() -> CommentMonitorManager:
        """
        Get or create a global comment monitoring manager
        
        Returns:
            CommentMonitorManager: The global comment monitoring manager
        """
        global comment_monitor
        if comment_monitor is None:
            comment_monitor = CommentMonitorManager(bot, dp)
        return comment_monitor
    
    @dp.message(UserStates.waiting_for_follow_up)
    async def handle_follow_up(message: types.Message, state: FSMContext):
        """Handle follow-up actions after issue creation or cancellation"""
        if message.text == "💭 Izoh qo'shish":
            # Fetch user's issues
            user_data = await state.get_data()
            contact_id = user_data.get('contact_id')
            
            if not contact_id:
                await message.answer("❌ Kontakt topilmadi. Iltimos, qaytadan raqamingizni yuboring.")
                await state.set_state(UserStates.waiting_for_contact)
                return
            
            # Find user's issues
            issues = await find_contact_issues(contact_id)
            
            if not issues:
                await message.answer("🔍 Sizda hozircha ochiq muammolar yo'q.")
                return
            
            # Create and send issues keyboard
            issues_keyboard = await create_issues_keyboard(issues)
            await message.answer(
                "🔢 Qaysi muammoga izoh qo'shmoqchisiz? Ro'yxatdan tanlang:",
                reply_markup=issues_keyboard
            )
            await state.set_state(UserStates.selecting_issue)
        elif message.text == "🆕 Yangi ariza yaratish":
            await message.answer(
                "Yangi arizangizni tasvirlab bering:",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.waiting_for_issue)
        else:
            # For any other message, offer options
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="💭 Izoh qo'shish")],
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            await message.answer(
                "Quyidagi amallardan birini tanlang:",
                reply_markup=follow_up_keyboard
            )

    @dp.message(UserStates.selecting_issue)
    async def handle_issue_selection(message: types.Message, state: FSMContext):
        """
        Handle user's issue selection for commenting
        """
        # Check if user wants to cancel
        if message.text.lower() == "cancel":
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="💭 Izoh qo'shish")],
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            await message.answer("❌ Amaliyot bekor qilindi.", reply_markup=follow_up_keyboard)
            await state.set_state(UserStates.waiting_for_follow_up)
            return
        
        # Parse issue ID from user's selection
        try:
            issue_id = int(message.text.split(']')[0].strip('[]'))
        except (ValueError, IndexError):
            await message.answer("❌ Noto'g'ri tanlov. Iltimos, ro'yxatdan tanlang.")
            return
        
        # Get contact ID from state
        user_data = await state.get_data()
        contact_id = user_data.get('contact_id')
        
        # Validate access to the issue
        if not await validate_issue_access(issue_id, contact_id):
            issues = await find_contact_issues(contact_id, with_closed=False)
            issues_keyboard = await create_issues_keyboard(issues) if issues else ReplyKeyboardRemove()
            await message.answer("❌ Siz faqat o'zingizning arizalaringizga izoh qo'shishingiz mumkin.", reply_markup=issues_keyboard)
            return
        
        # Save issue ID to state
        await state.update_data(selected_issue_id=issue_id)
        
        await message.answer(
            "✍️ Tanlangan muammoga qanday izoh qo'shmoqchisiz?\n\n"
            "📝 Matnli izoh qo‘shish uchun - xabar yuboring\n"
            "📷 Rasmlik izoh qo‘shish uchun - rasm yuboring\n"
            "🖼️ Rasm va matnli izoh qo‘shish uchun - rasm va izoh yuboring",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.set_state(UserStates.writing_comment)

    @dp.message(UserStates.writing_comment)
    async def submit_issue_comment(message: types.Message, state: FSMContext):
        """
        Submit the comment to OKDesk with optional photo attachment and confirm with the user
        """
        user_data = await state.get_data()
        issue_id = user_data.get('selected_issue_id')
        contact_id = user_data.get('contact_id')
        
        if not issue_id or not contact_id:
            await message.answer("❌ Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")
            await state.clear()
            return
        
        photo_data = None
        photo_filename = None
        comment_text = message.text or ""
        
        # Check if the message contains a photo
        if message.photo:
            # Get the largest photo (best quality)
            photo = message.photo[-1]
            
            try:
                # Download the photo
                photo_file = await bot.get_file(photo.file_id)
                photo_data_io = await bot.download_file(photo_file.file_path)
                photo_data = photo_data_io.read()
                
                # Generate a filename based on file_id
                photo_filename = f"photo_{photo.file_id}.jpg"
                
                logger.info(f"Downloaded photo: {photo_filename}, size: {len(photo_data)} bytes")
                
                # If message has caption, use it as comment text
                if message.caption:
                    comment_text = message.caption
                else:
                    comment_text = "Photo attachment"
            
            except Exception as e:
                logger.error(f"Error downloading photo: {e}")
                await message.answer("❌ Rasmni yuklashda xatolik yuz berdi. Iltimos, matnli izoh kiriting.")
                return
        
        # Add comment to issue with photo if available
        comment_result = await add_comment_to_issue(
            issue_id=issue_id, 
            comment_text=comment_text, 
            contact_id=contact_id,
            photo_data=photo_data,
            photo_filename=photo_filename
        )
        
        # Prepare follow-up keyboard
        follow_up_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="💭 Izoh qo'shish")],
                [KeyboardButton(text="🆕 Yangi ariza yaratish")]
            ],
            resize_keyboard=True
        )
        
        if comment_result:
            # Success message
            success_text = f"✅ Izoh #{issue_id} raqamli muammoga muvaffaqiyatli qo'shildi!"
            if photo_data:
                success_text += " (Rasm bilan)"
            await message.answer(success_text, reply_markup=follow_up_keyboard)
        else:
            await message.answer(
                "❌ Izoh qo'shishda xatolik yuz berdi.",
                reply_markup=follow_up_keyboard
            )
        
        await state.set_state(UserStates.waiting_for_follow_up)

    # We need to modify the approach to handle all message types
    # Let's register the handlers in a specific order

    # 1. First handle photos
    @dp.message(UserStates.writing_comment, F.photo)
    async def handle_photo_comment(message: types.Message, state: FSMContext):
        """Handle photo uploads during comment writing state"""
        user_data = await state.get_data()
        issue_id = user_data.get('selected_issue_id')
        contact_id = user_data.get('contact_id')
        
        if not issue_id or not contact_id:
            await message.answer("❌ Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")
            await state.clear()
            return
        
        # Get the highest quality photo
        photo = message.photo[-1]
        
        try:
            # Download the photo
            photo_file = await bot.get_file(photo.file_id)
            photo_data_io = await bot.download_file(photo_file.file_path)
            photo_data = photo_data_io.read()
            
            # Generate a filename based on file_id
            photo_filename = f"photo_{photo.file_id}.jpg"
            
            logger.info(f"Downloaded photo: {photo_filename}, size: {len(photo_data)} bytes")
            
            # Use caption as comment text, or default text if no caption
            comment_text = message.caption if message.caption else "Photo attachment"
            
            # Add comment to issue with photo
            comment_result = await add_comment_to_issue(
                issue_id=issue_id, 
                comment_text=comment_text, 
                contact_id=contact_id,
                photo_data=photo_data,
                photo_filename=photo_filename
            )
            
            # Prepare follow-up keyboard
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="💭 Izoh qo'shish")],
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            
            if comment_result:
                await message.answer(
                    f"✅ Rasm #{issue_id} raqamli muammoga muvaffaqiyatli qo'shildi!",
                    reply_markup=follow_up_keyboard
                )
            else:
                await message.answer(
                    "❌ Rasmni yuklashda xatolik yuz berdi.",
                    reply_markup=follow_up_keyboard
                )
            
            await state.set_state(UserStates.waiting_for_follow_up)
        
        except Exception as e:
            logger.error(f"Error processing photo: {e}")
            await message.answer(
                "❌ Rasmni yuklashda xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.",
                reply_markup=ReplyKeyboardRemove()
            )

    # 2. Then handle text messages
    @dp.message(UserStates.writing_comment, F.text)
    async def handle_text_comment(message: types.Message, state: FSMContext):
        """Handle text comments during comment writing state"""
        user_data = await state.get_data()
        issue_id = user_data.get('selected_issue_id')
        contact_id = user_data.get('contact_id')
        
        if not issue_id or not contact_id:
            await message.answer("❌ Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring.")
            await state.clear()
            return
        
        # Add comment to issue with text only
        comment_result = await add_comment_to_issue(
            issue_id=issue_id, 
            comment_text=message.text, 
            contact_id=contact_id
        )
        
        # Prepare follow-up keyboard
        follow_up_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="💭 Izoh qo'shish")],
                [KeyboardButton(text="🆕 Yangi ariza yaratish")]
            ],
            resize_keyboard=True
        )
        
        if comment_result:
            await message.answer(
                f"✅ Izoh #{issue_id} raqamli muammoga muvaffaqiyatli qo'shildi!",
                reply_markup=follow_up_keyboard
            )
        else:
            await message.answer(
                "❌ Izoh qo'shishda xatolik yuz berdi.",
                reply_markup=follow_up_keyboard
            )
        
        await state.set_state(UserStates.waiting_for_follow_up)

    # 3. Finally, handle any other message type
    @dp.message(UserStates.writing_comment)
    async def handle_unsupported_comment_type(message: types.Message, state: FSMContext):
        """Handle unsupported message types during comment writing"""
        await message.answer(
            "❌ Bu turdagi xabarlar qo'llab-quvvatlanmaydi. Iltimos, matn yoki rasmli xabar yuboring.",
            reply_markup=ReplyKeyboardRemove()
        )
    
    # Add general message handler for those not in any state
    @dp.message(Command("help"))
    async def cmd_help(message: types.Message):
        """Handle /help command"""
        help_text = (
            "Ushbu bot xizmat ko'rsatish markaziga buyurtmalar yaratishga imkon beradi.\n\n"
            "Buyruqlar:\n"
            "/start - Botni ishga tushirish\n"
            "/help - Yordam va ma'lumotlar\n"
            "/cancel - Buyurtmani bekor qilish\n\n"
            "Boshlash uchun /start buyrug'idan foydalaning va ko'rsatmalarga amal qiling."
        )
        await message.answer(help_text)
    
    @dp.message(Command("cancel"))
    async def cmd_cancel(message: types.Message, state: FSMContext):
        """Handle /cancel command in any state"""
        current_state = await state.get_state()
        if current_state is not None:
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            
            await state.clear()
            await message.answer(
                "Amal bekor qilindi.",
                reply_markup=follow_up_keyboard
            )
            await state.set_state(UserStates.waiting_for_follow_up)
        else:
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            
            await message.answer(
                "Bekor qilish uchun faol amal mavjud emas.",
                reply_markup=follow_up_keyboard
            )
            await state.set_state(UserStates.waiting_for_follow_up)
    
    # Default handler for any message in no specific state
    @dp.message()
    async def default_handler(message: types.Message, state: FSMContext):
        """Handle messages not covered by other handlers"""
        current_state = await state.get_state()
        
        if current_state is None:
            # If user is not in any state, move to follow-up state
            follow_up_keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="🆕 Yangi ariza yaratish")]
                ],
                resize_keyboard=True
            )
            
            await message.answer(
                "Yangi ariza yaratish uchun tugmani bosing.",
                reply_markup=follow_up_keyboard
            )
            await state.set_state(UserStates.waiting_for_follow_up)
    
    # Start the bot
    try:
        await dp.start_polling(bot)
    finally:
        # Cancel all comment monitoring tasks when bot shuts down
        for task in comment_tasks.values():
            task.cancel()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())