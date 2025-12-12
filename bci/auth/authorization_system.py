"""
Integrated Authorization System for Nexus AGI BCI
Restricts system access to Douglas Shane Davis only

Integrates:
- US Patent 3951134: Biometric EEG Authentication
- US Patent 6011991: Brain Wave Communication
- US Patent 5507291: Emotional State Detection
"""
import hashlib
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from bci.core.interfaces import BCIReading
from bci.auth.biometric_auth import BiometricEEGAuthenticator, AuthenticationResult
from bci.auth.brainwave_communication import BrainwaveCommunicationSystem, CommunicationIntent
from bci.auth.emotional_state import EmotionalStateDetector, EmotionalState


class AccessLevel(Enum):
    """System access levels"""
    NONE = "none"
    READ_ONLY = "read_only"
    STANDARD = "standard"
    ELEVATED = "elevated"
    FULL_CONTROL = "full_control"


class SecurityEvent(Enum):
    """Security event types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SESSION_EXPIRED = "session_expired"
    BIOMETRIC_MISMATCH = "biometric_mismatch"
    EMOTIONAL_ANOMALY = "emotional_anomaly"
    COMMUNICATION_DENIAL = "communication_denial"


@dataclass
class AuthorizedUser:
    """Authorized user definition"""
    user_id: str
    full_name: str
    access_level: AccessLevel
    require_biometric: bool = True
    require_emotional_check: bool = True
    allowed_operations: List[str] = None

    def __post_init__(self):
        if self.allowed_operations is None:
            self.allowed_operations = ["all"]


@dataclass
class SecurityAuditLog:
    """Security audit entry"""
    timestamp: str
    event_type: SecurityEvent
    user_id: Optional[str]
    success: bool
    details: Dict[str, Any]
    ip_address: Optional[str] = None


class NexusAGIAuthorizationSystem:
    """
    Master authorization system for Nexus AGI BCI

    SECURITY: Only Douglas Shane Davis is authorized for system access

    Integrates three patented technologies:
    1. Biometric EEG authentication (US Patent 3951134)
    2. Brain wave communication (US Patent 6011991)
    3. Emotional state detection (US Patent 5507291)
    """

    # AUTHORIZED USER - HARDCODED FOR SECURITY
    AUTHORIZED_USER_ID = "user_dsd_001"
    AUTHORIZED_FULL_NAME = "Douglas Shane Davis"
    AUTHORIZED_ACCESS_LEVEL = AccessLevel.FULL_CONTROL

    def __init__(self):
        # Initialize subsystems
        self.biometric_auth = BiometricEEGAuthenticator(
            authorized_user=self.AUTHORIZED_FULL_NAME
        )
        self.brainwave_comm = BrainwaveCommunicationSystem()
        self.emotional_detector = EmotionalStateDetector()

        # Define authorized user
        self.authorized_user = AuthorizedUser(
            user_id=self.AUTHORIZED_USER_ID,
            full_name=self.AUTHORIZED_FULL_NAME,
            access_level=self.AUTHORIZED_ACCESS_LEVEL,
            require_biometric=True,
            require_emotional_check=True,
            allowed_operations=["all"]
        )

        # Security state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.security_audit_log: List[SecurityAuditLog] = []
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_until: Dict[str, datetime] = {}

        # Configuration
        self.max_failed_attempts = 3
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=8)
        self.require_continuous_monitoring = True

        print("="*80)
        print("NEXUS AGI BCI AUTHORIZATION SYSTEM")
        print("="*80)
        print(f"Authorized User: {self.AUTHORIZED_FULL_NAME}")
        print(f"User ID: {self.AUTHORIZED_USER_ID}")
        print(f"Access Level: {self.AUTHORIZED_ACCESS_LEVEL.value}")
        print(f"")
        print(f"Patent Technologies Integrated:")
        print(f"  • US Patent 3951134: Biometric EEG Authentication")
        print(f"  • US Patent 6011991: Brain Wave Communication")
        print(f"  • US Patent 5507291: Emotional State Detection")
        print("="*80)

    def enroll_authorized_user(self, user_id: str, full_name: str,
                               calibration_readings: List[BCIReading]) -> bool:
        """
        Enroll the authorized user in the biometric system

        SECURITY: Only Douglas Shane Davis can be enrolled
        """
        print(f"\n[Authorization] 🔐 User Enrollment Request")
        print(f"[Authorization] Requested User: {full_name}")
        print(f"[Authorization] User ID: {user_id}")

        # CRITICAL SECURITY CHECK
        if full_name.upper() != self.AUTHORIZED_FULL_NAME.upper():
            self._log_security_event(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                user_id,
                False,
                {"reason": f"Unauthorized enrollment attempt by {full_name}"}
            )
            print(f"[Authorization] ❌ ENROLLMENT DENIED")
            print(f"[Authorization] Only {self.AUTHORIZED_FULL_NAME} is authorized")
            return False

        # Proceed with biometric enrollment
        success = self.biometric_auth.enroll_user(user_id, full_name, calibration_readings)

        if success:
            print(f"[Authorization] ✅ {full_name} enrolled successfully")
            self._log_security_event(
                SecurityEvent.LOGIN_SUCCESS,
                user_id,
                True,
                {"action": "enrollment", "calibration_samples": len(calibration_readings)}
            )
        else:
            print(f"[Authorization] ❌ Enrollment failed")

        return success

    def authenticate_user(self, user_id: str, reading: BCIReading) -> Optional[str]:
        """
        Authenticate user via multi-factor biometric authentication

        Returns session token if successful, None otherwise

        Authentication process:
        1. Check if user is Douglas Shane Davis
        2. Verify biometric EEG pattern
        3. Analyze emotional state for anomalies
        4. Request brain wave confirmation
        5. Generate secure session
        """
        print(f"\n[Authorization] 🔓 Authentication Request")
        print(f"[Authorization] User ID: {user_id}")
        print(f"[Authorization] Timestamp: {datetime.utcnow().isoformat()}")

        # Check lockout
        if self._is_locked_out(user_id):
            lockout_time = self.lockout_until[user_id]
            remaining = (lockout_time - datetime.utcnow()).total_seconds() / 60
            print(f"[Authorization] 🔒 Account locked out for {remaining:.1f} more minutes")
            return None

        # STEP 1: Verify user is authorized
        if user_id != self.AUTHORIZED_USER_ID:
            self._record_failed_attempt(user_id)
            self._log_security_event(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                user_id,
                False,
                {"reason": f"User {user_id} is not authorized"}
            )
            print(f"[Authorization] ❌ UNAUTHORIZED USER")
            print(f"[Authorization] Only {self.AUTHORIZED_FULL_NAME} ({self.AUTHORIZED_USER_ID}) is authorized")
            return None

        # STEP 2: Biometric EEG authentication
        print(f"[Authorization] Step 1/4: Biometric EEG authentication...")
        auth_result = self.biometric_auth.authenticate(reading, user_id)

        if not auth_result.success:
            self._record_failed_attempt(user_id)
            self._log_security_event(
                SecurityEvent.BIOMETRIC_MISMATCH,
                user_id,
                False,
                {"reason": auth_result.failure_reason, "confidence": auth_result.confidence}
            )
            print(f"[Authorization] ❌ Biometric authentication failed")
            return None

        print(f"[Authorization] ✅ Biometric match: {auth_result.confidence:.2%}")

        # STEP 3: Emotional state analysis
        print(f"[Authorization] Step 2/4: Emotional state analysis...")
        emotional_profile = self.emotional_detector.detect_emotional_state(reading)

        if emotional_profile:
            print(f"[Authorization] Emotional State: {emotional_profile.primary_emotion.value}")
            print(f"[Authorization] Stress Level: {emotional_profile.stress_level:.2%}")
            print(f"[Authorization] Positivity: {emotional_profile.positivity:.2%}")

            # Check for emotional anomalies
            anomaly = self.emotional_detector.detect_emotional_anomaly(emotional_profile)
            if anomaly:
                print(f"[Authorization] ⚠️  {anomaly}")
                self._log_security_event(
                    SecurityEvent.EMOTIONAL_ANOMALY,
                    user_id,
                    True,
                    {"anomaly": anomaly, "emotion": emotional_profile.primary_emotion.value}
                )

            # Elevated stress check
            if emotional_profile.stress_level > 0.85:
                print(f"[Authorization] ⚠️  Very high stress detected - proceed with caution")

        # STEP 4: Brain wave communication confirmation
        print(f"[Authorization] Step 3/4: Brain wave communication...")
        comm_message = self.brainwave_comm.decode_message(reading, user_id)

        if comm_message:
            print(f"[Authorization] Brain Wave Intent: {comm_message.intent.value}")
            print(f"[Authorization] Confidence: {comm_message.confidence:.2%}")

            # Check if user is denying access
            if comm_message.intent in [CommunicationIntent.NO, CommunicationIntent.DENY, CommunicationIntent.CANCEL]:
                print(f"[Authorization] ❌ User denied access via brain wave")
                self._log_security_event(
                    SecurityEvent.COMMUNICATION_DENIAL,
                    user_id,
                    False,
                    {"intent": comm_message.intent.value}
                )
                return None

        # STEP 5: Generate secure session
        print(f"[Authorization] Step 4/4: Creating secure session...")
        session_token = auth_result.session_token

        # Create enhanced session with all monitoring data
        self.active_sessions[session_token] = {
            "user_id": user_id,
            "full_name": self.AUTHORIZED_FULL_NAME,
            "access_level": self.AUTHORIZED_ACCESS_LEVEL.value,
            "start_time": datetime.utcnow(),
            "expires": datetime.utcnow() + self.session_timeout,
            "biometric_confidence": auth_result.confidence,
            "emotional_state": emotional_profile.primary_emotion.value if emotional_profile else "unknown",
            "last_activity": datetime.utcnow(),
            "operations_performed": 0
        }

        # Reset failed attempts
        self.failed_attempts[user_id] = 0

        # Log success
        self._log_security_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id,
            True,
            {
                "biometric_confidence": auth_result.confidence,
                "emotional_state": emotional_profile.primary_emotion.value if emotional_profile else None,
                "session_token": session_token[:16]
            }
        )

        print(f"\n[Authorization] ✅ AUTHENTICATION SUCCESSFUL")
        print(f"[Authorization] Welcome, {self.AUTHORIZED_FULL_NAME}")
        print(f"[Authorization] Access Level: {self.AUTHORIZED_ACCESS_LEVEL.value}")
        print(f"[Authorization] Session Token: {session_token[:16]}...")
        print(f"[Authorization] Session expires: {self.active_sessions[session_token]['expires']}")

        return session_token

    def verify_access(self, session_token: str, operation: str = "general") -> bool:
        """
        Verify if session token has access to perform operation
        """
        if session_token not in self.active_sessions:
            print(f"[Authorization] ❌ Invalid session token")
            return False

        session = self.active_sessions[session_token]

        # Check expiration
        if datetime.utcnow() > session["expires"]:
            print(f"[Authorization] ❌ Session expired")
            self._log_security_event(
                SecurityEvent.SESSION_EXPIRED,
                session["user_id"],
                False,
                {"session_duration": str(datetime.utcnow() - session["start_time"])}
            )
            del self.active_sessions[session_token]
            return False

        # Update last activity
        session["last_activity"] = datetime.utcnow()
        session["operations_performed"] += 1

        # Check access level for operation
        access_level = AccessLevel(session["access_level"])
        if access_level == AccessLevel.FULL_CONTROL:
            return True  # Full control can do everything

        # Operation-specific checks would go here
        return True

    def continuous_monitoring(self, session_token: str, reading: BCIReading) -> bool:
        """
        Continuous biometric and emotional monitoring during active session
        Detects if unauthorized person is attempting to use the system
        """
        if session_token not in self.active_sessions:
            return False

        session = self.active_sessions[session_token]
        user_id = session["user_id"]

        # Re-verify biometric
        auth_result = self.biometric_auth.authenticate(reading, user_id)

        if auth_result.confidence < 0.70:  # Lower threshold for continuous monitoring
            print(f"[Authorization] ⚠️  Biometric confidence dropped: {auth_result.confidence:.2%}")
            print(f"[Authorization] Possible unauthorized user detected")

            if auth_result.confidence < 0.50:  # Critical threshold
                print(f"[Authorization] 🚨 CRITICAL: Session terminated due to biometric mismatch")
                self.revoke_session(session_token, "biometric_confidence_critical")
                return False

        # Monitor emotional state
        emotional_profile = self.emotional_detector.detect_emotional_state(reading)
        if emotional_profile:
            anomaly = self.emotional_detector.detect_emotional_anomaly(emotional_profile)
            if anomaly:
                print(f"[Authorization] ⚠️  {anomaly}")

        return True

    def revoke_session(self, session_token: str, reason: str = "manual"):
        """Revoke/terminate a session"""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]
            user_id = session["user_id"]

            print(f"[Authorization] 🔒 Session revoked for {session['full_name']}")
            print(f"[Authorization] Reason: {reason}")

            self._log_security_event(
                SecurityEvent.SESSION_EXPIRED,
                user_id,
                True,
                {"reason": reason, "operations_performed": session["operations_performed"]}
            )

            del self.active_sessions[session_token]

    def get_session_info(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get information about active session"""
        return self.active_sessions.get(session_token)

    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is currently locked out"""
        if user_id in self.lockout_until:
            if datetime.utcnow() < self.lockout_until[user_id]:
                return True
            else:
                del self.lockout_until[user_id]
        return False

    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt"""
        self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1

        if self.failed_attempts[user_id] >= self.max_failed_attempts:
            self.lockout_until[user_id] = datetime.utcnow() + self.lockout_duration
            print(f"[Authorization] 🔒 User {user_id} locked out for {self.lockout_duration}")

    def _log_security_event(self, event_type: SecurityEvent, user_id: Optional[str],
                           success: bool, details: Dict[str, Any]):
        """Log security event to audit log"""
        log_entry = SecurityAuditLog(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            user_id=user_id,
            success=success,
            details=details
        )
        self.security_audit_log.append(log_entry)

    def get_security_audit(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security audit logs"""
        recent_logs = self.security_audit_log[-limit:]
        return [
            {
                "timestamp": log.timestamp,
                "event": log.event_type.value,
                "user_id": log.user_id,
                "success": log.success,
                "details": log.details
            }
            for log in recent_logs
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "authorized_user": self.AUTHORIZED_FULL_NAME,
            "active_sessions": len(self.active_sessions),
            "failed_attempts_tracked": len(self.failed_attempts),
            "locked_out_users": len(self.lockout_until),
            "total_security_events": len(self.security_audit_log),
            "biometric_auth_stats": self.biometric_auth.get_authentication_stats(),
            "emotional_detector_stats": self.emotional_detector.get_statistics(),
            "brainwave_comm_stats": self.brainwave_comm.get_statistics()
        }

    def print_status(self):
        """Print formatted system status"""
        status = self.get_system_status()

        print("\n" + "="*80)
        print("NEXUS AGI AUTHORIZATION SYSTEM - STATUS")
        print("="*80)

        print(f"\n👤 AUTHORIZED USER:")
        print(f"  Name: {status['authorized_user']}")
        print(f"  User ID: {self.AUTHORIZED_USER_ID}")
        print(f"  Access Level: {self.AUTHORIZED_ACCESS_LEVEL.value}")

        print(f"\n🔐 SESSION STATUS:")
        print(f"  Active Sessions: {status['active_sessions']}")
        print(f"  Failed Attempts: {status['failed_attempts_tracked']}")
        print(f"  Locked Out Users: {status['locked_out_users']}")

        print(f"\n📊 BIOMETRIC AUTHENTICATION:")
        bio_stats = status['biometric_auth_stats']
        print(f"  Total Attempts: {bio_stats['total_attempts']}")
        print(f"  Successful: {bio_stats['successful']}")
        print(f"  Failed: {bio_stats['failed']}")
        print(f"  Success Rate: {bio_stats['success_rate']:.1%}")

        print(f"\n🧠 EMOTIONAL STATE DETECTION:")
        emo_stats = status['emotional_detector_stats']
        if emo_stats.get('status') != 'no_data':
            print(f"  Total Readings: {emo_stats['total_readings']}")
            print(f"  Baseline Established: {emo_stats['baseline_established']}")
            print(f"  Current Emotion: {emo_stats['current_emotion']}")

        print(f"\n📡 BRAIN WAVE COMMUNICATION:")
        comm_stats = status['brainwave_comm_stats']
        print(f"  Total Messages: {comm_stats['total_messages']}")
        print(f"  Pending Messages: {comm_stats['pending_messages']}")
        print(f"  Avg Confidence: {comm_stats['average_confidence']:.2%}")

        print("\n" + "="*80)
