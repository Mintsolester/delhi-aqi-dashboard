"""
Notification System
Handles email/SMS/push notifications for AQI alerts
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime
import json

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    logger.warning("SendGrid not installed")

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not installed")

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationService:
    """
    Manages user notifications for AQI alerts
    Supports email, SMS, and push notifications
    """
    
    def __init__(self):
        """Initialize notification service"""
        self.sendgrid_client = None
        self.twilio_client = None
        self.subscribers_file = config.DATA_DIR / "subscribers.json"
        self.subscribers = self._load_subscribers()
        
        # Initialize clients
        self._init_sendgrid()
        self._init_twilio()
    
    def _init_sendgrid(self):
        """Initialize SendGrid client"""
        if SENDGRID_AVAILABLE and config.SENDGRID_API_KEY:
            try:
                self.sendgrid_client = SendGridAPIClient(config.SENDGRID_API_KEY)
                logger.info("SendGrid initialized")
            except Exception as e:
                logger.error(f"SendGrid initialization failed: {str(e)}")
    
    def _init_twilio(self):
        """Initialize Twilio client"""
        if TWILIO_AVAILABLE and config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN:
            try:
                self.twilio_client = TwilioClient(
                    config.TWILIO_ACCOUNT_SID,
                    config.TWILIO_AUTH_TOKEN
                )
                logger.info("Twilio initialized")
            except Exception as e:
                logger.error(f"Twilio initialization failed: {str(e)}")
    
    def _load_subscribers(self) -> Dict:
        """Load subscriber data"""
        if self.subscribers_file.exists():
            try:
                with open(self.subscribers_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading subscribers: {str(e)}")
        
        return {'email': [], 'sms': [], 'preferences': {}}
    
    def _save_subscribers(self):
        """Save subscriber data"""
        try:
            with open(self.subscribers_file, 'w') as f:
                json.dump(self.subscribers, f, indent=2)
            logger.info("Subscribers saved")
        except Exception as e:
            logger.error(f"Error saving subscribers: {str(e)}")
    
    def subscribe_email(self,
                       email: str,
                       locations: List[str],
                       threshold_aqi: int = 200,
                       vulnerable_groups: Optional[List[str]] = None):
        """
        Subscribe user to email alerts
        
        Args:
            email: User email address
            locations: List of locations to monitor
            threshold_aqi: Alert threshold
            vulnerable_groups: User's vulnerable group memberships
        """
        subscriber = {
            'email': email,
            'locations': locations,
            'threshold_aqi': threshold_aqi,
            'vulnerable_groups': vulnerable_groups or [],
            'subscribed_at': datetime.now().isoformat(),
            'active': True
        }
        
        # Check if already subscribed
        existing = next(
            (s for s in self.subscribers['email'] if s['email'] == email),
            None
        )
        
        if existing:
            # Update existing subscription
            existing.update(subscriber)
            logger.info(f"Updated subscription for {email}")
        else:
            # Add new subscription
            self.subscribers['email'].append(subscriber)
            logger.info(f"New email subscription: {email}")
        
        self._save_subscribers()
    
    def subscribe_sms(self,
                     phone: str,
                     locations: List[str],
                     threshold_aqi: int = 200):
        """
        Subscribe user to SMS alerts
        
        Args:
            phone: Phone number (with country code)
            locations: List of locations to monitor
            threshold_aqi: Alert threshold
        """
        subscriber = {
            'phone': phone,
            'locations': locations,
            'threshold_aqi': threshold_aqi,
            'subscribed_at': datetime.now().isoformat(),
            'active': True
        }
        
        existing = next(
            (s for s in self.subscribers['sms'] if s['phone'] == phone),
            None
        )
        
        if existing:
            existing.update(subscriber)
            logger.info(f"Updated SMS subscription for {phone}")
        else:
            self.subscribers['sms'].append(subscriber)
            logger.info(f"New SMS subscription: {phone}")
        
        self._save_subscribers()
    
    def unsubscribe_email(self, email: str):
        """Unsubscribe email"""
        subscriber = next(
            (s for s in self.subscribers['email'] if s['email'] == email),
            None
        )
        
        if subscriber:
            subscriber['active'] = False
            self._save_subscribers()
            logger.info(f"Unsubscribed: {email}")
            return True
        
        return False
    
    def send_email_alert(self,
                        to_email: str,
                        subject: str,
                        content: str,
                        html_content: Optional[str] = None) -> bool:
        """
        Send email alert
        
        Args:
            to_email: Recipient email
            subject: Email subject
            content: Plain text content
            html_content: Optional HTML content
            
        Returns:
            True if sent successfully
        """
        if not self.sendgrid_client:
            logger.warning("SendGrid not available, logging email instead")
            logger.info(f"Email to {to_email}: {subject}")
            return False
        
        try:
            message = Mail(
                from_email=config.EMAIL_FROM,
                to_emails=to_email,
                subject=subject,
                plain_text_content=content,
                html_content=html_content
            )
            
            response = self.sendgrid_client.send(message)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent to {to_email}")
                return True
            else:
                logger.error(f"Email send failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Email send error: {str(e)}")
            return False
    
    def send_sms_alert(self,
                      to_phone: str,
                      message: str) -> bool:
        """
        Send SMS alert
        
        Args:
            to_phone: Recipient phone number
            message: SMS message (max 160 chars recommended)
            
        Returns:
            True if sent successfully
        """
        if not self.twilio_client:
            logger.warning("Twilio not available, logging SMS instead")
            logger.info(f"SMS to {to_phone}: {message}")
            return False
        
        try:
            response = self.twilio_client.messages.create(
                body=message,
                from_=config.SMS_FROM,
                to=to_phone
            )
            
            logger.info(f"SMS sent to {to_phone}, SID: {response.sid}")
            return True
            
        except Exception as e:
            logger.error(f"SMS send error: {str(e)}")
            return False
    
    def notify_forecast_alert(self,
                             location: str,
                             current_aqi: float,
                             forecast_aqi: float,
                             peak_time: str,
                             category: str):
        """
        Send forecast alerts to subscribed users
        
        Args:
            location: Location name
            current_aqi: Current AQI value
            forecast_aqi: Forecasted peak AQI
            peak_time: Time of peak pollution
            category: AQI category
        """
        # Email alerts
        for subscriber in self.subscribers['email']:
            if not subscriber.get('active', True):
                continue
            
            if location in subscriber['locations']:
                if forecast_aqi >= subscriber['threshold_aqi']:
                    
                    subject = f"⚠️ Air Quality Alert - {location}"
                    
                    content = f"""
                    Air Quality Forecast Alert
                    
                    Location: {location}
                    Current AQI: {int(current_aqi)}
                    Forecasted Peak AQI: {int(forecast_aqi)}
                    Category: {category}
                    Expected Peak Time: {peak_time}
                    
                    Please take necessary precautions:
                    - Limit outdoor activities
                    - Wear N95/N99 masks if going outside
                    - Keep windows closed
                    - Use air purifiers if available
                    
                    Stay safe!
                    Delhi AQI Dashboard
                    """
                    
                    html_content = f"""
                    <html>
                        <body style="font-family: Arial, sans-serif;">
                            <h2 style="color: #ff0000;">⚠️ Air Quality Alert</h2>
                            <p><strong>Location:</strong> {location}</p>
                            <p><strong>Current AQI:</strong> {int(current_aqi)}</p>
                            <p><strong>Forecasted Peak AQI:</strong> 
                               <span style="color: #ff0000; font-size: 24px;">{int(forecast_aqi)}</span>
                            </p>
                            <p><strong>Category:</strong> {category}</p>
                            <p><strong>Expected Peak Time:</strong> {peak_time}</p>
                            
                            <h3>Recommendations:</h3>
                            <ul>
                                <li>Limit outdoor activities</li>
                                <li>Wear N95/N99 masks if going outside</li>
                                <li>Keep windows closed</li>
                                <li>Use air purifiers if available</li>
                            </ul>
                            
                            <p>Stay safe!</p>
                            <p><em>Delhi AQI Dashboard</em></p>
                        </body>
                    </html>
                    """
                    
                    self.send_email_alert(
                        subscriber['email'],
                        subject,
                        content,
                        html_content
                    )
        
        # SMS alerts
        for subscriber in self.subscribers['sms']:
            if not subscriber.get('active', True):
                continue
            
            if location in subscriber['locations']:
                if forecast_aqi >= subscriber['threshold_aqi']:
                    
                    sms_message = (
                        f"AQI Alert - {location}: "
                        f"Expected to reach {int(forecast_aqi)} ({category}) "
                        f"around {peak_time}. Take precautions!"
                    )
                    
                    self.send_sms_alert(subscriber['phone'], sms_message)
    
    def get_subscriber_count(self) -> Dict:
        """Get subscriber statistics"""
        active_email = sum(
            1 for s in self.subscribers['email'] 
            if s.get('active', True)
        )
        
        active_sms = sum(
            1 for s in self.subscribers['sms']
            if s.get('active', True)
        )
        
        return {
            'email_subscribers': active_email,
            'sms_subscribers': active_sms,
            'total_active': active_email + active_sms
        }


if __name__ == "__main__":
    # Example usage
    service = NotificationService()
    
    # Subscribe user
    service.subscribe_email(
        email="user@example.com",
        locations=["Anand Vihar", "Connaught Place"],
        threshold_aqi=150,
        vulnerable_groups=["asthmatic"]
    )
    
    # Send test alert
    service.notify_forecast_alert(
        location="Anand Vihar",
        current_aqi=120,
        forecast_aqi=250,
        peak_time="6:00 PM",
        category="Very Poor"
    )
    
    # Get stats
    stats = service.get_subscriber_count()
    print(f"\nSubscriber Stats: {stats}")
