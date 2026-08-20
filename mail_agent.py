from langchain.tools import tool
from langchain.tools import Tool
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def _mail_tool(input_data: dict) -> str:
    """
    Expects a single dict param named 'input_data'. Example:
      mail_tool({"input_data": {
         "analysis_file": "detections/analysis.txt",
         "image_file": "detections/1.png"
      }})
    """
    analysis_file = input_data.get("analysis_file", "")
    image_file = input_data.get("image_file", "")

    if not os.path.exists(analysis_file):
        return f"Analysis file not found: {analysis_file}"

    with open(analysis_file, "r") as f:
        email_body = f.read().strip()

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    USERNAME = ""
    PASSWORD = ""

    msg = MIMEMultipart()
    msg["Subject"] = "DeepSeek Analysis Report"
    msg["From"] = USERNAME
    msg["To"] = ""
    msg.attach(MIMEText(email_body, "plain"))

    if image_file and os.path.isfile(image_file):
        try:
            with open(image_file, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename={os.path.basename(image_file)}'
            )
            msg.attach(part)
        except Exception as e:
            return f"Error attaching {image_file}: {e}"
    else:
        return f"No image file found at '{image_file}'. Email not sent."

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(USERNAME, PASSWORD)
            server.send_message(msg)
    except Exception as e:
        return f"Failed to send email: {str(e)}"

    return f"Email sent successfully to {msg['To']}."

mail_tool = Tool(name="mail_tool", func=_mail_tool, return_direct=True)
