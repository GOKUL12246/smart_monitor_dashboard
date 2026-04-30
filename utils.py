import smtplib
import os
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import traceback
from datetime import datetime

# ======= CONFIG =======
GMAIL_SENDER = "g17670714@gmail.com"
GMAIL_PASSWORD = "hexe syjd dfyv wbal"  # App Password (not regular password)
GMAIL_RECEIVER = "g17670714@gmail.com"

TWILIO_SID = "ACf93bf342b2329038428531fc5433950a"
TWILIO_AUTH = "b82ed646f75d652e60ffbb2e2dcae601"
TWILIO_NUMBER = "+19787966432"
RECEIVER_NUMBER = "+916369067138"
# =====================

def send_email_alert(message, subject=None, attachment_path=None, attachment_bytes=None, attachment_filename=None):
    """Send email alert via Gmail SMTP.

    Args:
        message: Alert body text.
        subject: Optional email subject.
        attachment_path: Optional file path to attach (legacy mode).
        attachment_bytes: Optional in-memory file bytes to attach.
        attachment_filename: Optional file name for in-memory attachment.
    """
    try:
        print("[EMAIL] Starting email send...")
        # Don't print subject with emojis to avoid encoding issues
        print(f"[EMAIL] Subject length: {len(subject) if subject else 0} chars")
        print(f"[EMAIL] Message preview: {message[:100] if len(message) > 100 else message}...")
        
        # Use multipart/mixed so text + HTML + optional attachment can be sent together.
        msg = MIMEMultipart()
        msg["Subject"] = subject or "Temple Crowd Alert!"
        msg["From"] = GMAIL_SENDER
        msg["To"] = GMAIL_RECEIVER
        
        # Plain text version
        text_content = f"""
TEMPLE CROWD MONITORING ALERT
=============================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

This is an automated alert from the Temple Crowd Monitoring System.
        """
        
        # HTML version
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h2 style="color: #e74c3c; border-bottom: 2px solid #e74c3c; padding-bottom: 10px;">
                    🚨 Temple Crowd Monitoring Alert
                </h2>
                <p style="color: #666; font-size: 14px;">
                    <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                <div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin: 15px 0;">
                    <p style="margin: 0; font-size: 16px; color: #856404;">
                        <strong>{message}</strong>
                    </p>
                </div>
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="color: #999; font-size: 12px; text-align: center;">
                    This is an automated alert from the Temple Crowd Monitoring System.
                </p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        if attachment_bytes is not None:
            filename = attachment_filename or "missing_person_live.jpg"
            ctype, _ = mimetypes.guess_type(filename)
            maintype, subtype = ("application", "octet-stream")
            if ctype and "/" in ctype:
                maintype, subtype = ctype.split("/", 1)

            part = MIMEBase(maintype, subtype)
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)
            print(f"[EMAIL] Attached in-memory file: {filename}")
        elif attachment_path:
            if os.path.exists(attachment_path) and os.path.isfile(attachment_path):
                ctype, _ = mimetypes.guess_type(attachment_path)
                maintype, subtype = ("application", "octet-stream")
                if ctype and "/" in ctype:
                    maintype, subtype = ctype.split("/", 1)

                with open(attachment_path, "rb") as f:
                    part = MIMEBase(maintype, subtype)
                    part.set_payload(f.read())

                encoders.encode_base64(part)
                filename = os.path.basename(attachment_path)
                part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
                msg.attach(part)
                print(f"[EMAIL] Attached file: {filename}")
            else:
                print(f"[EMAIL] Attachment not found, sending without file: {attachment_path}")
        
        print("[EMAIL] Connecting to Gmail SMTP server...")
        # Send via Gmail SMTP - optimized for fast delivery
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=5) as server:
            server.ehlo()
            print("[EMAIL] Starting TLS...")
            server.starttls()
            server.ehlo()
            print(f"[EMAIL] Logging in as {GMAIL_SENDER}...")
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            print(f"[EMAIL] Sending message to {GMAIL_RECEIVER}...")
            server.send_message(msg)
            print("[EMAIL] Message sent to Gmail servers!")
        
        print(f"[SUCCESS] Email alert sent successfully to {GMAIL_RECEIVER}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"[ERROR] Email auth failed - Check Gmail App Password: {e}")
        print("   [TIP] Create App Password at https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")
        traceback.print_exc()
        return False

def test_email():
    """Test email functionality"""
    return send_email_alert("TEST ALERT: This is a test message from Temple Crowd Monitor.", 
                           "🔔 Test Alert - Temple Monitor")

def send_sms_alert(message):
    try:
        # SMS alerts disabled - Twilio credentials need updating
        print("⚠️  SMS alerts currently disabled (update Twilio credentials to enable)")
        return
        
        # Uncomment below when you have valid Twilio credentials
        # client = Client(TWILIO_SID, TWILIO_AUTH)
        # client.messages.create(
        #     body=message,
        #     from_=TWILIO_NUMBER,
        #     to=RECEIVER_NUMBER
        # )
        # print("📱 SMS alert sent")
    except Exception as e:
        print("❌ SMS failed:", e)
        traceback.print_exc()


def send_access_granted_alert(person_name, confidence, location="Face Access Control", timestamp=None):
    """
    Send access granted alert via email when face recognition grants access.
    
    Args:
        person_name: Name of person granted access
        confidence: Match confidence score (0-100)
        location: Zone/location name where access was granted
        timestamp: Optional timestamp (auto-generated if not provided)
    
    Returns:
        bool: True if alert sent successfully, False otherwise
    """
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        confidence_pct = min(100, max(0, confidence))
        
        # Prepare email subject and message
        subject = f"✅ ACCESS GRANTED: {person_name}"
        
        message = f"""
FACE ACCESS CONTROL - ACCESS GRANTED
=====================================
Time: {timestamp}
Location: {location}
Confidence: {confidence_pct:.1f}%

Person Name: {person_name}
Status: ACCESS GRANTED
Confidence Level: {confidence_pct:.1f}%

Access Decision: ✅ APPROVED
Entry Authorized at {location}

Face Recognition Match: {confidence_pct:.1f}%
System: Smart Monitor Dashboard - Face Access Control
"""
        
        # Send email alert
        print(f"[ACCESS_GRANT] Sending access granted alert for {person_name}...")
        success = send_email_alert(
            message=message,
            subject=subject
        )
        
        if success:
            print(f"[✅ ACCESS] Email notification sent for {person_name} at {location}")
        
        return success
        
    except Exception as e:
        print(f"❌ Access granted alert failed: {e}")
        traceback.print_exc()
        return False


def send_missing_person_alert(name, confidence, location, image_array=None, extra_message=""):
    """
    Send missing person identification alert via email and audio.
    
    Args:
        name: Missing person's name
        confidence: Match confidence score (0-100)
        location: Detection location/zone name
        image_array: Optional numpy array of detection frame (in-memory, not saved)
        extra_message: Additional message details
    
    Returns:
        bool: True if alert sent successfully, False otherwise
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence_pct = min(100, max(0, confidence))
        
        # Prepare email subject and message
        subject = f"🚨 MISSING PERSON DETECTED: {name}"
        
        message = f"""
MISSING PERSON IDENTIFICATION ALERT
====================================
Time: {timestamp}
Location: {location}
Confidence: {confidence_pct:.1f}%

Missing Person Name: {name}
Status: IDENTIFIED in camera view

Zone: {location}
Confidence Level: {confidence_pct:.1f}%

{extra_message}

IMMEDIATE ACTION REQUIRED!
- Alert authorities immediately
- Verify identity with security personnel
- Keep person in sight until authorities arrive
"""
        
        # Convert image to JPEG bytes for email attachment (in-memory, no disk save)
        attachment_bytes = None
        attachment_filename = None
        
        if image_array is not None:
            try:
                import cv2
                # Encode frame as JPEG in memory
                _, jpeg_bytes = cv2.imencode('.jpg', image_array)
                if jpeg_bytes is not None:
                    attachment_bytes = jpeg_bytes.tobytes()
                    attachment_filename = f"missing_person_{name}_{timestamp.replace(':', '-')}.jpg"
                    print(f"[ALERT] Image prepared: {len(attachment_bytes)} bytes")
            except Exception as e:
                print(f"[WARN] Could not prepare image attachment: {e}")
        
        # Send email alert
        print(f"[ALERT] Sending missing person alert for {name}...")
        send_email_alert(
            message=message,
            subject=subject,
            attachment_bytes=attachment_bytes,
            attachment_filename=attachment_filename
        )
        
        # Play audio alert on speaker
        try:
            from voice_alerts import speak_custom, AlertPriority
            alert_msg = f"ALERT! Missing person {name} detected at {location} with {confidence_pct:.0f} percent confidence"
            speak_custom(alert_msg, priority=AlertPriority.CRITICAL)
            print(f"[ALERT] Audio alert queued: {alert_msg}")
        except Exception as e:
            print(f"[WARN] Could not play audio alert: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Missing person alert failed: {e}")
        traceback.print_exc()
        return False

