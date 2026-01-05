"""
SMTP Email Notifier for Bridge Transactions
Sends email notifications for transaction updates and validations
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMTPNotifier:
    """
    SMTP Email Notifier
    Sends transaction notifications via email
    """

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None
    ):
        """
        Initialize SMTP notifier

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password or app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv('SMTP_EMAIL')
        self.sender_password = sender_password or os.getenv('SMTP_PASSWORD')

        logger.info(f"SMTP Notifier initialized: {smtp_server}:{smtp_port}")

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        html: bool = False
    ) -> bool:
        """
        Send email notification

        Args:
            recipient: Recipient email address
            subject: Email subject
            body: Email body
            html: Whether body is HTML formatted

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password:
            logger.warning("SMTP credentials not configured, skipping email")
            return False

        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient

            # Add body
            mime_type = "html" if html else "plain"
            message.attach(MIMEText(body, mime_type))

            # Create secure SSL context
            context = ssl.create_default_context()

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient, message.as_string())

            logger.info(f"✓ Email sent to {recipient}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def notify_bridge_initiated(
        self,
        recipient: str,
        bridge_id: str,
        source_address: str,
        destination_address: str,
        amount_btc: float,
        amount_eth: float
    ) -> bool:
        """Send notification when bridge transfer is initiated"""
        subject = f"Bridge Transfer Initiated - {bridge_id}"

        body = f"""
Bridge Transfer Initiated
========================

Bridge ID: {bridge_id}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Transfer Details:
- Source Chain: Bitcoin
- Destination Chain: Ethereum
- Source Address: {source_address}
- Destination Address: {destination_address}
- Amount (BTC): {amount_btc}
- Amount (WBTC): {amount_eth}

Status: Initiated

Your Bitcoin tokens will be bridged to Ethereum as wrapped BTC (WBTC).
You will receive confirmation emails as the transaction progresses.

---
Nexus AGI Bridge System
        """

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #2c3e50;">🌉 Bridge Transfer Initiated</h2>

    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <p><strong>Bridge ID:</strong> {bridge_id}</p>
        <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>

    <h3 style="color: #34495e;">Transfer Details</h3>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #ecf0f1;">
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Source Chain</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">Bitcoin</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Destination Chain</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">Ethereum</td>
        </tr>
        <tr style="background-color: #ecf0f1;">
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Source Address</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7; font-family: monospace; font-size: 12px;">{source_address}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Destination Address</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7; font-family: monospace; font-size: 12px;">{destination_address}</td>
        </tr>
        <tr style="background-color: #ecf0f1;">
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Amount (BTC)</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">{amount_btc}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Amount (WBTC)</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">{amount_eth}</td>
        </tr>
    </table>

    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0;"><strong>Status:</strong> ✅ Initiated</p>
    </div>

    <p style="color: #7f8c8d; font-size: 14px;">
        Your Bitcoin tokens will be bridged to Ethereum as wrapped BTC (WBTC).
        You will receive confirmation emails as the transaction progresses.
    </p>

    <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 30px 0;">
    <p style="color: #95a5a6; font-size: 12px; text-align: center;">
        Nexus AGI Bridge System
    </p>
</body>
</html>
        """

        return self.send_email(recipient, subject, html_body, html=True)

    def notify_btc_confirmed(
        self,
        recipient: str,
        bridge_id: str,
        btc_tx_hash: str,
        confirmations: int
    ) -> bool:
        """Send notification when Bitcoin transaction is confirmed"""
        subject = f"Bitcoin Transaction Confirmed - {bridge_id}"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #27ae60;">✅ Bitcoin Transaction Confirmed</h2>

    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <p><strong>Bridge ID:</strong> {bridge_id}</p>
        <p><strong>BTC Transaction:</strong> <code style="background-color: #ecf0f1; padding: 5px;">{btc_tx_hash}</code></p>
        <p><strong>Confirmations:</strong> {confirmations}</p>
    </div>

    <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0;"><strong>Next Step:</strong> Broadcasting to Ethereum network...</p>
    </div>

    <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 30px 0;">
    <p style="color: #95a5a6; font-size: 12px; text-align: center;">
        Nexus AGI Bridge System
    </p>
</body>
</html>
        """

        return self.send_email(recipient, subject, html_body, html=True)

    def notify_eth_broadcasted(
        self,
        recipient: str,
        bridge_id: str,
        eth_tx_hash: str,
        amount_eth: float,
        destination_address: str
    ) -> bool:
        """Send notification when Ethereum transaction is broadcasted"""
        subject = f"Ethereum Transaction Broadcasted - {bridge_id}"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #3498db;">📡 Broadcasting to Ethereum Network</h2>

    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <p><strong>Bridge ID:</strong> {bridge_id}</p>
        <p><strong>ETH Transaction:</strong> <code style="background-color: #ecf0f1; padding: 5px;">{eth_tx_hash}</code></p>
        <p><strong>Amount:</strong> {amount_eth} WBTC</p>
        <p><strong>Destination:</strong> <code style="background-color: #ecf0f1; padding: 5px;">{destination_address}</code></p>
    </div>

    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0;"><strong>Status:</strong> ⏳ Waiting for Ethereum confirmations...</p>
    </div>

    <p style="color: #7f8c8d; font-size: 14px;">
        View on Etherscan: <a href="https://etherscan.io/tx/{eth_tx_hash}">https://etherscan.io/tx/{eth_tx_hash}</a>
    </p>

    <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 30px 0;">
    <p style="color: #95a5a6; font-size: 12px; text-align: center;">
        Nexus AGI Bridge System
    </p>
</body>
</html>
        """

        return self.send_email(recipient, subject, html_body, html=True)

    def notify_transfer_completed(
        self,
        recipient: str,
        bridge_id: str,
        btc_tx_hash: str,
        eth_tx_hash: str,
        amount_btc: float,
        amount_eth: float,
        destination_address: str
    ) -> bool:
        """Send notification when bridge transfer is completed"""
        subject = f"🎉 Bridge Transfer Completed - {bridge_id}"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #27ae60;">🎉 Bridge Transfer Completed Successfully!</h2>

    <div style="background-color: #d4edda; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #28a745;">
        <p style="margin: 0; font-size: 18px;"><strong>✅ Transfer Complete</strong></p>
    </div>

    <h3 style="color: #34495e;">Transfer Summary</h3>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="background-color: #ecf0f1;">
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Bridge ID</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">{bridge_id}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>BTC Amount</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">{amount_btc} BTC</td>
        </tr>
        <tr style="background-color: #ecf0f1;">
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>WBTC Received</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7;">{amount_eth} WBTC</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Destination</strong></td>
            <td style="padding: 10px; border: 1px solid #bdc3c7; font-family: monospace; font-size: 12px;">{destination_address}</td>
        </tr>
    </table>

    <h3 style="color: #34495e;">Transaction Hashes</h3>
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p><strong>Bitcoin:</strong><br>
        <a href="https://blockstream.info/tx/{btc_tx_hash}" style="font-family: monospace; font-size: 12px;">{btc_tx_hash}</a></p>

        <p><strong>Ethereum:</strong><br>
        <a href="https://etherscan.io/tx/{eth_tx_hash}" style="font-family: monospace; font-size: 12px;">{eth_tx_hash}</a></p>
    </div>

    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0;"><strong>✅ Both transactions fully validated and confirmed!</strong></p>
    </div>

    <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 30px 0;">
    <p style="color: #95a5a6; font-size: 12px; text-align: center;">
        Nexus AGI Bridge System
    </p>
</body>
</html>
        """

        return self.send_email(recipient, subject, html_body, html=True)

    def notify_validation_complete(
        self,
        recipient: str,
        validation_summary: dict
    ) -> bool:
        """Send notification with validation summary"""
        subject = "Bridge Validation Report"

        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <h2 style="color: #2c3e50;">📊 Bridge Validation Report</h2>

    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h3>Validation Summary</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Total Transactions</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{validation_summary.get('total_transactions', 0)}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Completed</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; color: #27ae60;">{validation_summary.get('completed', 0)}</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Pending</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; color: #f39c12;">{validation_summary.get('pending', 0)}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Failed</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7; color: #e74c3c;">{validation_summary.get('failed', 0)}</td>
            </tr>
            <tr style="background-color: #ecf0f1;">
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Bitcoin Validated</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{validation_summary.get('bitcoin_validated', 0)}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Ethereum Validated</strong></td>
                <td style="padding: 10px; border: 1px solid #bdc3c7;">{validation_summary.get('ethereum_validated', 0)}</td>
            </tr>
        </table>
    </div>

    <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0;"><strong>✅ All transactions validated on both Bitcoin and Ethereum networks</strong></p>
    </div>

    <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 30px 0;">
    <p style="color: #95a5a6; font-size: 12px; text-align: center;">
        Nexus AGI Bridge System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    </p>
</body>
</html>
        """

        return self.send_email(recipient, subject, html_body, html=True)


if __name__ == "__main__":
    # Test SMTP notifier
    print("SMTP Notifier Module")
    print("Configure SMTP_EMAIL and SMTP_PASSWORD environment variables to use.")
