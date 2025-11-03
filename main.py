import sqlite3
import bcrypt
import jwt
import datetime
import secrets
import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import uvicorn
import nest_asyncio
from pydantic import BaseModel, EmailStr, field_validator
from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from collections import Counter
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import warnings
import threading
import time
import socket
from pyngrok import ngrok
import random

# --- Configuration ---
warnings.filterwarnings('ignore')
DetectorFactory.seed = 0
nest_asyncio.apply()

JWT_EXPIRY_HOURS = 24
SUPPORTED_LANGUAGES = ['English', 'Hindi']
DATABASE_NAME = 'health_chatbot_enhanced.db'
SECRET_KEY = secrets.token_hex(32)

# Add after JWT_EXPIRY_HOURS
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Optional - for real AI
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # Optional - for real AI

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    confirmPassword: str
    name: str
    age: Optional[int] = None
    languagePreference: str = 'English'

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v

    @field_validator('confirmPassword')
    @classmethod
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters long')
        return v.strip()

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class HealthLog(BaseModel):
    steps: Optional[int] = None
    calories: Optional[int] = None
    sleep_hours: Optional[float] = None
    water_intake: Optional[float] = None
    mood: Optional[str] = None
    notes: Optional[str] = None

class JournalEntry(BaseModel):
    title: str
    content: str

class GoalCreate(BaseModel):
    goal_type: str
    target_value: float
    description: Optional[str] = None

class DietPlanRequest(BaseModel):
    diet_preference: str
    allergies: Optional[str] = None
    health_goal: str
    calories_target: Optional[int] = 2000

class CommunityPost(BaseModel):
    title: str
    content: str
    anonymous: bool = False

class LLMService:
    """Service for integrating with LLM APIs for personalized content generation"""

    def __init__(self, provider="mock"):
        self.provider = provider  # "mock", "gemini", "openai"
        self.mock_mode = provider == "mock"

    def generate_diet_plan(self, user_profile: Dict, health_logs: List[Dict]) -> Dict:
        """Generate personalized diet plan using LLM"""

        if self.mock_mode:
            return self._mock_generate_diet_plan(user_profile, health_logs)

        # Real LLM integration would go here
        prompt = self._build_diet_prompt(user_profile, health_logs)
        # response = call_llm_api(prompt)
        # return parse_diet_response(response)

        return self._mock_generate_diet_plan(user_profile, health_logs)

    def _build_diet_prompt(self, user_profile: Dict, health_logs: List[Dict]) -> str:
        """Build comprehensive prompt for LLM"""
        avg_calories = np.mean([log.get('calories', 2000) for log in health_logs if log.get('calories')])
        avg_water = np.mean([log.get('water_intake', 2) for log in health_logs if log.get('water_intake')])
        mood_trend = Counter([log.get('mood') for log in health_logs if log.get('mood')]).most_common(1)

        prompt = f"""
        Generate a personalized 7-day diet plan for a user with the following profile:

        User Profile:
        - Dietary Preference: {user_profile.get('diet_preference', 'vegetarian')}
        - Health Goal: {user_profile.get('health_goal', 'maintenance')}
        - Allergies: {user_profile.get('allergies', 'None')}
        - Target Calories: {user_profile.get('calories_target', 2000)} per day
        - Location: India (include local cuisine options)

        Recent Health Data (Last 30 days):
        - Average Daily Calories: {avg_calories:.0f}
        - Average Water Intake: {avg_water:.1f}L
        - Dominant Mood: {mood_trend[0][0] if mood_trend else 'neutral'}

        Please provide:
        1. Daily meal plans (breakfast, lunch, dinner, 2 snacks) with specific portions
        2. Nutritional breakdown (calories, protein, carbs, fats) for each meal
        3. Local cuisine alternatives for each meal
        4. Hydration recommendations
        5. Specific tips based on their health goal and mood patterns

        Format as JSON with keys: meals, nutrition, tips, alternatives
        """

        return prompt

    def _mock_generate_diet_plan(self, user_profile: Dict, health_logs: List[Dict]) -> Dict:
        """Enhanced mock generation with AI-like personalization"""

        preference = user_profile.get('diet_preference', 'vegetarian')
        goal = user_profile.get('health_goal', 'maintenance')
        calories_target = user_profile.get('calories_target', 2000)

        # Analyze health logs for personalization
        avg_mood = 'good'
        if health_logs:
            moods = [log.get('mood') for log in health_logs if log.get('mood')]
            if moods:
                mood_counts = Counter(moods)
                avg_mood = mood_counts.most_common(1)[0][0] if mood_counts else 'good'

        # Generate contextual meal plans
        base_meals = {
            'breakfast': self._generate_breakfast(preference, calories_target, avg_mood),
            'lunch': self._generate_lunch(preference, calories_target, goal),
            'dinner': self._generate_dinner(preference, calories_target, goal),
            'snacks': self._generate_snacks(preference, avg_mood)
        }

        return {
            'meals': base_meals,
            'nutrition': self._calculate_nutrition(base_meals, calories_target),
            'tips': self._generate_personalized_tips(user_profile, health_logs),
            'local_alternatives': self._generate_local_alternatives(preference),
            'reasoning': self._explain_plan_reasoning(user_profile, health_logs)
        }

    def _generate_breakfast(self, preference: str, calories: int, mood: str) -> List[str]:
        """Generate contextual breakfast options"""
        energy_boost = ["Oats with bananas and honey", "Whole grain toast with peanut butter"]
        mood_boosters = ["Greek yogurt with berries and walnuts", "Smoothie bowl with chia seeds"]

        if mood in ['bad', 'sad']:
            return mood_boosters
        return energy_boost

    def _generate_lunch(self, preference: str, calories: int, goal: str) -> List[str]:
        """Generate contextual lunch options"""
        if goal == 'weight_loss':
            return [
                "Quinoa salad with grilled vegetables (350 cal)",
                "Lentil soup with whole wheat chapati (380 cal)"
            ]
        elif goal == 'muscle_gain':
            return [
                "Paneer tikka with brown rice (550 cal)",
                "Chickpea curry with quinoa (580 cal)"
            ]
        return [
            "Mixed vegetable curry with rice (450 cal)",
            "Whole wheat pasta with vegetables (430 cal)"
        ]

    def _generate_dinner(self, preference: str, calories: int, goal: str) -> List[str]:
        """Generate contextual dinner options"""
        if goal == 'weight_loss':
            return [
                "Grilled tofu with steamed vegetables (320 cal)",
                "Vegetable stir-fry with cauliflower rice (350 cal)"
            ]
        return [
            "Dal makhani with brown rice (450 cal)",
            "Vegetable biryani with raita (480 cal)"
        ]

    def _generate_snacks(self, preference: str, mood: str) -> List[str]:
        """Generate contextual snacks"""
        return [
            "Apple slices with almond butter (150 cal)",
            "Roasted chickpeas (120 cal)",
            "Mixed nuts and dried fruits (180 cal)"
        ]

    def _calculate_nutrition(self, meals: Dict, target_calories: int) -> Dict:
        """Calculate nutritional breakdown"""
        return {
            'daily_calories': target_calories,
            'protein_g': int(target_calories * 0.25 / 4),
            'carbs_g': int(target_calories * 0.45 / 4),
            'fats_g': int(target_calories * 0.30 / 9),
            'fiber_g': 30,
            'water_liters': 3.0
        }

    def _generate_personalized_tips(self, profile: Dict, logs: List[Dict]) -> List[str]:
        """Generate AI-style personalized tips"""
        tips = []

        if logs:
            avg_water = np.mean([log.get('water_intake', 0) for log in logs if log.get('water_intake')])
            if avg_water < 2.5:
                tips.append("💧 Your water intake is below optimal. Try setting hourly reminders to sip water.")

        goal = profile.get('health_goal', 'maintenance')
        if goal == 'weight_loss':
            tips.append("🥗 For weight loss, focus on high-volume, low-calorie foods like vegetables and lean proteins.")
        elif goal == 'muscle_gain':
            tips.append("💪 For muscle gain, ensure you're eating 0.8-1g protein per pound of body weight daily.")

        tips.append("🍽️ Practice mindful eating: chew slowly and stop when 80% full.")

        return tips

    def _generate_local_alternatives(self, preference: str) -> Dict:
        """Generate local Indian alternatives"""
        return {
            'breakfast': ['Poha with vegetables', 'Upma with sambar', 'Idli with coconut chutney'],
            'lunch': ['Rajma chawal', 'Chole bhature (healthier version)', 'Palak paneer with roti'],
            'dinner': ['Khichdi with ghee', 'Vegetable pulao with raita', 'Masoor dal with rice'],
            'snacks': ['Masala peanuts', 'Roasted makhana', 'Fruit chaat']
        }

    def _explain_plan_reasoning(self, profile: Dict, logs: List[Dict]) -> str:
        """Explain why this plan was generated"""
        goal = profile.get('health_goal', 'maintenance')
        calories = profile.get('calories_target', 2000)

        reasoning = f"This plan is optimized for {goal} with a target of {calories} calories per day. "

        if logs:
            avg_mood = 'neutral'
            moods = [log.get('mood') for log in logs if log.get('mood')]
            if moods:
                mood_counts = Counter(moods)
                avg_mood = mood_counts.most_common(1)[0][0]

            if avg_mood in ['bad', 'sad']:
                reasoning += "I've included mood-boosting foods rich in omega-3s and B-vitamins. "

        reasoning += "All meals are balanced with complex carbs, lean proteins, and healthy fats for sustained energy."

        return reasoning

    def generate_fitness_plan(self, user_profile: Dict, health_logs: List[Dict]) -> Dict:
        """Generate personalized fitness plan using LLM"""

        if self.mock_mode:
            return self._mock_generate_fitness_plan(user_profile, health_logs)

        # Real LLM integration
        prompt = self._build_fitness_prompt(user_profile, health_logs)
        return self._mock_generate_fitness_plan(user_profile, health_logs)

    def _mock_generate_fitness_plan(self, user_profile: Dict, health_logs: List[Dict]) -> Dict:
        """Enhanced fitness plan generation"""
        level = user_profile.get('fitness_level', 'beginner')
        goal = user_profile.get('goal', 'maintenance')

        # Analyze activity patterns from logs
        avg_steps = 0
        if health_logs:
            steps = [log.get('steps', 0) for log in health_logs if log.get('steps')]
            avg_steps = np.mean(steps) if steps else 0

        weekly_plan = self._generate_weekly_workout(level, goal, avg_steps)

        return {
            'weekly_plan': weekly_plan,
            'progression': self._generate_progression_plan(level),
            'recovery_tips': self._generate_recovery_tips(level),
            'reasoning': f"Based on your {level} fitness level and current average of {avg_steps:.0f} daily steps, this plan gradually builds intensity."
        }

    def _generate_weekly_workout(self, level: str, goal: str, avg_steps: float) -> Dict:
        """Generate contextual weekly workout"""
        if level == 'beginner':
            return {
                'monday': '20 min brisk walk + 10 min stretching',
                'tuesday': '15 min bodyweight exercises (squats, lunges, push-ups - 3 sets)',
                'wednesday': 'Rest or gentle yoga (20 min)',
                'thursday': '25 min cycling or swimming',
                'friday': '15 min HIIT workout (beginners)',
                'saturday': '30 min outdoor activity (hiking, sports)',
                'sunday': 'Active rest - stretching or walking'
            }
        # Add intermediate and advanced plans
        return {}

    def _generate_progression_plan(self, level: str) -> List[str]:
        """Generate progression milestones"""
        return [
            "Week 1-2: Focus on form and consistency",
            "Week 3-4: Increase duration by 5-10%",
            "Week 5-6: Add intensity or weight",
            "Week 7-8: Challenge yourself with advanced variations"
        ]

    def _generate_recovery_tips(self, level: str) -> List[str]:
        """Generate recovery recommendations"""
        return [
            "Ensure 7-9 hours of sleep for optimal recovery",
            "Stay hydrated: drink water before, during, and after workouts",
            "Stretch for 5-10 minutes post-workout",
            "Take rest days seriously - muscles grow during rest"
        ]
# ============================================================================
# PROACTIVE NUDGE SYSTEM
# ============================================================================

class ProactiveNudgeScheduler:
    """Background system that analyzes trends and sends proactive recommendations"""

    def __init__(self, db_manager, health_chatbot):
        self.db_manager = db_manager
        self.health_chatbot = health_chatbot
        self.running = False
        self.check_interval = 3600  # Check every hour
        self.nudge_history = {}  # Track sent nudges to avoid spam

    def start(self):
        """Start the background scheduler"""
        self.running = True
        thread = threading.Thread(target=self._run_scheduler, daemon=True)
        thread.start()
        print("✅ Proactive Nudge Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False

    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._check_all_users()
            except Exception as e:
                print(f"Error in nudge scheduler: {e}")

            time.sleep(self.check_interval)

    def _check_all_users(self):
        """Check all active users for nudge opportunities"""
        with sqlite3.connect(self.db_manager.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, email, name, language_preference FROM users')
            users = cursor.fetchall()

        for user in users:
            user_id, email, name, lang = user
            nudges = self._analyze_user_trends(user_id, lang)

            if nudges:
                self._send_nudges(user_id, nudges)

    def _analyze_user_trends(self, user_id: int, language: str = 'English') -> List[Dict]:
        """Analyze user data and generate appropriate nudges"""
        nudges = []

        # Get recent logs (last 7 days)
        logs = self.db_manager.get_health_logs(user_id, days=7)

        if len(logs) < 3:
            return nudges

        # Check sleep pattern
        sleep_data = [log.get('sleep_hours', 0) for log in logs if log.get('sleep_hours')]
        if len(sleep_data) >= 3:
            avg_sleep = np.mean(sleep_data)
            if avg_sleep < 6.5:
                nudges.append({
                    'type': 'sleep_concern',
                    'priority': 'high',
                    'message': f"😴 You've averaged {avg_sleep:.1f} hours of sleep recently. This is below the recommended 7-9 hours. Poor sleep affects mood, focus, and physical health.",
                    'action': "Try our sleep meditation tonight",
                    'language': language
                })

        # Check water intake
        water_data = [log.get('water_intake', 0) for log in logs if log.get('water_intake')]
        if len(water_data) >= 3:
            avg_water = np.mean(water_data)
            if avg_water < 2.0:
                nudges.append({
                    'type': 'hydration',
                    'priority': 'medium',
                    'message': f"💧 Your water intake has been around {avg_water:.1f}L/day. Aim for 3-4L for optimal hydration and energy.",
                    'action': "Set hourly water reminders",
                    'language': language
                })

        # Check mood patterns
        moods = [log.get('mood') for log in logs if log.get('mood')]
        if len(moods) >= 3:
            negative_moods = sum(1 for m in moods if m in ['bad', 'sad', 'angry'])
            if negative_moods >= 2:
                nudges.append({
                    'type': 'mood_support',
                    'priority': 'high',
                    'message': "🌟 I notice you've been having some tough days recently. Remember, it's okay to not be okay. Small steps matter.",
                    'action': "Try journaling or talking to someone you trust",
                    'language': language
                })

        # Check activity levels
        steps_data = [log.get('steps', 0) for log in logs if log.get('steps')]
        if len(steps_data) >= 3:
            avg_steps = np.mean(steps_data)
            if avg_steps < 5000:
                nudges.append({
                    'type': 'activity',
                    'priority': 'medium',
                    'message': f"🚶 Your daily steps have been around {avg_steps:.0f}. Even a 10-minute walk can boost your mood and energy!",
                    'action': "Take a short walk today",
                    'language': language
                })

        # Check streak maintenance
        if len(logs) >= 7 and not any(nudge['type'] == 'celebration' for nudge in nudges):
            nudges.append({
                'type': 'celebration',
                'priority': 'low',
                'message': f"🎉 Amazing! You've logged health data for 7 consecutive days! You're building a powerful habit.",
                'action': "Keep the streak going!",
                'language': language
            })

        return nudges

    def _send_nudges(self, user_id: int, nudges: List[Dict]):
        """Send nudges to user (store for retrieval via API)"""
        # Check if we've sent similar nudges recently (don't spam)
        current_time = datetime.datetime.now()

        for nudge in nudges:
            nudge_key = f"{user_id}_{nudge['type']}"
            last_sent = self.nudge_history.get(nudge_key)

            # Only send if we haven't sent this type in the last 24 hours
            if last_sent is None or (current_time - last_sent).total_seconds() > 86400:
                # Store nudge in database
                with sqlite3.connect(self.db_manager.db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''INSERT INTO nudges
                        (user_id, type, priority, message, action, language, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (user_id, nudge['type'], nudge['priority'], nudge['message'],
                         nudge['action'], nudge['language'], current_time))
                    conn.commit()

                self.nudge_history[nudge_key] = current_time
                print(f"📬 Sent {nudge['type']} nudge to user {user_id}")

# ============================================================================
# MICRO HABIT STACKER
# ============================================================================

class MicroHabitStacker:
    """Help users build tiny habits onto existing routines"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.common_anchors = [
            "Making morning coffee",
            "Brushing teeth",
            "Taking a shower",
            "Sitting down at desk",
            "Eating lunch",
            "Commuting home",
            "Watching TV in evening",
            "Getting into bed"
        ]

    def suggest_micro_habit(self, goal_type: str, anchor_habit: str = None) -> Dict:
        """Suggest a micro habit to stack onto existing routine"""

        suggestions = {
            'water': {
                'Making morning coffee': "After I turn on the coffee machine, I will drink one glass of water",
                'Brushing teeth': "After I brush my teeth, I will drink a glass of water",
                'Eating lunch': "Before I start eating lunch, I will drink a glass of water"
            },
            'meditation': {
                'Making morning coffee': "While my coffee brews, I will take 3 deep breaths",
                'Sitting down at desk': "After I sit at my desk, I will close my eyes for 1 minute",
                'Getting into bed': "After I get into bed, I will do 2 minutes of breathing exercises"
            },
            'exercise': {
                'Brushing teeth': "After I brush my teeth, I will do 10 squats",
                'Making morning coffee': "While my coffee brews, I will do 5 push-ups or wall push-ups",
                'Commuting home': "After I get home, I will take a 5-minute walk around the block"
            },
            'sleep': {
                'Watching TV in evening': "After I turn off the TV, I will dim the lights",
                'Brushing teeth': "After I brush my teeth at night, I will read for 10 minutes",
                'Getting into bed': "After I get into bed, I will put my phone away"
            }
        }

        if anchor_habit and goal_type in suggestions and anchor_habit in suggestions[goal_type]:
            return {
                'anchor': anchor_habit,
                'micro_habit': suggestions[goal_type][anchor_habit],
                'why_it_works': "Tiny habits are easier to start and more likely to stick when linked to existing routines."
            }

        # Return all suggestions for the goal type
        if goal_type in suggestions:
            return {
                'suggestions': [
                    {'anchor': anchor, 'micro_habit': habit}
                    for anchor, habit in suggestions[goal_type].items()
                ],
                'why_it_works': "Pick an anchor habit you do reliably every day, then stack a tiny new habit onto it."
            }

        return {'message': 'No suggestions available for this goal type yet.'}

import threading
db_lock = threading.RLock()
# ============================================================================
# ENHANCED DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self, db_name: str = DATABASE_NAME):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Initialize database with proper table creation"""
        # Use check_same_thread=False for SQLite in multi-threaded environment
        with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
            conn.isolation_level = None  # Autocommit mode
            cursor = conn.cursor()

            # Users table
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                secret_key TEXT NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                language_preference TEXT DEFAULT 'English',
                theme_preference TEXT DEFAULT 'dark',
                wellness_points INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # Chat history
            cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                intent TEXT,
                entity TEXT,
                detected_language TEXT,
                sentiment TEXT,
                emotion TEXT,
                emotion_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # Health logs
            cursor.execute('''CREATE TABLE IF NOT EXISTS health_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                log_date DATE NOT NULL,
                steps INTEGER,
                calories INTEGER,
                sleep_hours REAL,
                water_intake REAL,
                mood TEXT,
                heart_rate INTEGER,
                stress_level INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # Journal entries
            cursor.execute('''CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                mood TEXT,
                sentiment TEXT,
                emotion_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # Wellness goals
            cursor.execute('''CREATE TABLE IF NOT EXISTS wellness_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                goal_type TEXT NOT NULL,
                target_value REAL NOT NULL,
                current_value REAL DEFAULT 0,
                current_streak INTEGER DEFAULT 0,
                best_streak INTEGER DEFAULT 0,
                description TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # ⭐ MISSING TABLE - Goal progress tracking
            cursor.execute('''CREATE TABLE IF NOT EXISTS goal_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                date DATE NOT NULL,
                value REAL NOT NULL,
                completed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(goal_id, date),
                FOREIGN KEY (goal_id) REFERENCES wellness_goals (id)
            )''')

            # Achievements
            cursor.execute('''CREATE TABLE IF NOT EXISTS achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                badge_name TEXT NOT NULL,
                badge_icon TEXT,
                points_awarded INTEGER DEFAULT 0,
                earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # ⭐ MISSING TABLE - Community posts
            cursor.execute('''CREATE TABLE IF NOT EXISTS community_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                anonymous INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # Diet plans
            cursor.execute('''CREATE TABLE IF NOT EXISTS diet_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                diet_preference TEXT,
                allergies TEXT,
                health_goal TEXT,
                calories_target INTEGER,
                plan_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # Nudges
            cursor.execute('''CREATE TABLE IF NOT EXISTS nudges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                priority TEXT NOT NULL,
                message TEXT NOT NULL,
                action TEXT,
                language TEXT DEFAULT 'English',
                read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # User connections (Wellness Buddies)
            cursor.execute('''CREATE TABLE IF NOT EXISTS user_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                buddy_id INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accepted_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (buddy_id) REFERENCES users (id)
            )''')

            # Points store
            cursor.execute('''CREATE TABLE IF NOT EXISTS store_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                cost_points INTEGER NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # User purchases
            cursor.execute('''CREATE TABLE IF NOT EXISTS user_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                purchased_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (item_id) REFERENCES store_items (id)
            )''')

            # Challenge groups
            cursor.execute('''CREATE TABLE IF NOT EXISTS challenge_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                goal_type TEXT NOT NULL,
                target_value REAL NOT NULL,
                created_by INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_date DATE,
                FOREIGN KEY (created_by) REFERENCES users (id)
            )''')

            # Group members
            cursor.execute('''CREATE TABLE IF NOT EXISTS group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES challenge_groups (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # Sleep stages
            cursor.execute('''CREATE TABLE IF NOT EXISTS sleep_stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER NOT NULL,
                light_hours REAL,
                deep_hours REAL,
                rem_hours REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (log_id) REFERENCES health_logs (id)
            )''')

            # Micro habits
            cursor.execute('''CREATE TABLE IF NOT EXISTS micro_habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                anchor_habit TEXT NOT NULL,
                new_habit TEXT NOT NULL,
                goal_id INTEGER,
                completed_days INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            # CBT thought records
            cursor.execute('''CREATE TABLE IF NOT EXISTS cbt_thought_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                journal_entry_id INTEGER,
                negative_thought TEXT NOT NULL,
                thought_pattern TEXT,
                reframe_prompt TEXT,
                user_reframe TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (journal_entry_id) REFERENCES journal_entries (id)
            )''')

            # Wellness soundscapes
            cursor.execute('''CREATE TABLE IF NOT EXISTS wellness_soundscapes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                emotion TEXT NOT NULL,
                soundscape_config TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')

            conn.commit()

            # Populate store items if empty
            cursor.execute('SELECT COUNT(*) FROM store_items')
            if cursor.fetchone()[0] == 0:
                store_items = [
                    ('Dark Mode Theme', 'Sleek dark theme for night owls', 100, 'theme'),
                    ('Ocean Theme', 'Calming blue ocean theme', 150, 'theme'),
                    ('Premium Meditation: Deep Sleep', '30-minute guided meditation', 200, 'meditation'),
                    ('CBT Workbook', 'Interactive CBT exercises', 300, 'resource'),
                    ('Wellness Badge Pack', '10 exclusive badges', 250, 'badge'),
                    ('Personal AI Coach (1 week)', 'Enhanced AI recommendations', 500, 'premium')
                ]
                cursor.executemany(
                    'INSERT INTO store_items (name, description, cost_points, category) VALUES (?, ?, ?, ?)',
                    store_items
                )
                conn.commit()

            print("✓ Database initialized successfully with all tables")

    def create_user(self, email: str, password: str, name: str, age: Optional[int] = None,
                language_preference: str = 'English') -> Dict[str, Any]:
        """Create user with welcome bonus"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    cursor = conn.cursor()

                    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                    secret_key = secrets.token_hex(32)

                    cursor.execute('''INSERT INTO users (email, password_hash, secret_key, name, age, language_preference)
                        VALUES (?, ?, ?, ?, ?, ?)''', (email, password_hash, secret_key, name, age, language_preference))

                    user_id = cursor.lastrowid

                    # Award welcome bonus in same transaction
                    cursor.execute(
                        'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                        (50, user_id)
                    )

                    conn.commit()

                return {"success": True, "user_id": user_id, "message": "User created successfully"}
        except sqlite3.IntegrityError:
            return {"success": False, "message": "Email already registered"}
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}





    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
                user_data = cursor.fetchone()

                if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data[2]):
                    return {
                        'id': user_data[0],
                        'email': user_data[1],
                        'secret_key': user_data[3],
                        'name': user_data[4],
                        'age': user_data[5],
                        'language_preference': user_data[6],
                        'theme_preference': user_data[7],
                        'wellness_points': user_data[8]
                    }
        except Exception as e:
            print(f"Authentication error: {str(e)}")
        return None

    def save_chat_message(self, user_id: int, user_message: str, bot_response: str,
                          intent: str = None, entity: str = None, detected_language: str = None,
                          sentiment: str = None, emotion: str = None, emotion_score: float = None):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO chat_history
                    (user_id, user_message, bot_response, intent, entity, detected_language, sentiment, emotion, emotion_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (user_id, user_message, bot_response, intent, entity, detected_language, sentiment, emotion, emotion_score))
                conn.commit()
        except Exception as e:
            print(f"Error saving chat: {str(e)}")

    def save_health_log(self, user_id: int, log_data: Dict[str, Any]):
        """Save health log with proper error handling"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    cursor = conn.cursor()
                    log_date = datetime.date.today().isoformat()

                    cursor.execute('''INSERT INTO health_logs
                        (user_id, log_date, steps, calories, sleep_hours, water_intake, mood, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                        (user_id, log_date, log_data.get('steps'), log_data.get('calories'),
                        log_data.get('sleep_hours'), log_data.get('water_intake'),
                        log_data.get('mood'), log_data.get('notes')))

                    # Award points in same transaction
                    cursor.execute(
                        'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                        (10, user_id)
                    )

                    conn.commit()

                    # Check for streak milestone
                    cursor.execute('''SELECT COUNT(DISTINCT log_date) FROM health_logs
                        WHERE user_id = ? AND log_date >= date('now', '-7 days')''', (user_id,))
                    streak = cursor.fetchone()[0]

                    if streak == 7:
                        cursor.execute(
                            'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                            (50, user_id)
                        )
                        conn.commit()

                    return {"success": True, "message": "Health log saved successfully", "points_earned": 10}
        except Exception as e:
            print(f"Error saving health log: {str(e)}")
            return {"success": False, "message": f"Error saving health log: {str(e)}"}



    def get_health_logs(self, user_id: int, days: int = 30) -> List[Dict]:
        """Get health logs with proper error handling"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''SELECT log_date, steps, calories, sleep_hours, water_intake, mood, notes
                        FROM health_logs WHERE user_id = ? AND log_date >= date('now', '-' || ? || ' days')
                        ORDER BY log_date DESC''', (user_id, days))

                    logs = [dict(row) for row in cursor.fetchall()]
                    return logs
        except Exception as e:
            print(f"Error getting health logs: {str(e)}")
            return []


    def save_journal_entry(self, user_id: int, title: str, content: str, mood: str = None,
                       sentiment: str = None, emotion_score: float = None):
        """Save journal entry with proper error handling"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''INSERT INTO journal_entries
                        (user_id, title, content, mood, sentiment, emotion_score)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                        (user_id, title, content, mood, sentiment, emotion_score))

                    entry_id = cursor.lastrowid

                    # Award points in same transaction
                    cursor.execute(
                        'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                        (15, user_id)
                    )

                    conn.commit()

                    return {"success": True, "message": "Journal entry saved", "entry_id": entry_id}
        except Exception as e:
            print(f"Error saving journal: {str(e)}")
            return {"success": False, "message": f"Error saving journal: {str(e)}"}




    def get_journal_entries(self, user_id: int, days: int = 30) -> List[Dict]:
        """Get journal entries with proper error handling"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''SELECT id, title, content, mood, sentiment, created_at
                        FROM journal_entries WHERE user_id = ? AND created_at >= datetime('now', '-' || ? || ' days')
                        ORDER BY created_at DESC LIMIT 30''', (user_id, days))
                    entries = [dict(row) for row in cursor.fetchall()]
                    return entries
        except Exception as e:
            print(f"Error getting journal entries: {str(e)}")
            return []

    def save_goal(self, user_id: int, goal_type: str, target_value: float, description: str = None):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO wellness_goals (user_id, goal_type, target_value, description)
                    VALUES (?, ?, ?, ?)''', (user_id, goal_type, target_value, description))
                conn.commit()
                return {"success": True, "goal_id": cursor.lastrowid}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_user_goals(self, user_id: int) -> List[Dict]:
        """Get user goals with proper error handling"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''SELECT id, goal_type, target_value, current_value,
                        current_streak, best_streak, description, is_active, created_at
                        FROM wellness_goals WHERE user_id = ? AND is_active = 1
                        ORDER BY created_at DESC''', (user_id,))
                    goals = [dict(row) for row in cursor.fetchall()]
                    return goals
        except Exception as e:
            print(f"Error getting goals: {str(e)}")
            return []

    def update_goal_progress(self, goal_id: int, value: float):
        """Update goal progress - FIXED to avoid database locking"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    cursor = conn.cursor()
                    today = datetime.date.today().isoformat()

                    # Check if progress already logged today
                    cursor.execute('SELECT id FROM goal_progress WHERE goal_id = ? AND date = ?',
                                  (goal_id, today))
                    if cursor.fetchone():
                        return {"success": False, "message": "Progress for this goal already logged today."}

                    # Check yesterday's completion
                    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
                    cursor.execute('SELECT completed FROM goal_progress WHERE goal_id = ? AND date = ?',
                                  (goal_id, yesterday))
                    yesterday_result = cursor.fetchone()
                    was_completed_yesterday = yesterday_result[0] if yesterday_result else 0

                    # Get goal info
                    cursor.execute('SELECT target_value, user_id FROM wellness_goals WHERE id = ?', (goal_id,))
                    goal_info = cursor.fetchone()

                    if not goal_info:
                        return {"success": False, "message": "Goal not found"}

                    target_value, user_id = goal_info[0], goal_info[1]
                    is_completed = 1 if value >= target_value else 0

                    # Insert progress
                    cursor.execute('''INSERT INTO goal_progress (goal_id, date, value, completed)
                        VALUES (?, ?, ?, ?)''', (goal_id, today, value, is_completed))

                    # Update goal streak
                    if is_completed:
                        if was_completed_yesterday:
                            cursor.execute('''UPDATE wellness_goals SET current_value = ?,
                                current_streak = current_streak + 1 WHERE id = ?''',
                                (value, goal_id))
                        else:
                            cursor.execute('''UPDATE wellness_goals SET current_value = ?,
                                current_streak = 1 WHERE id = ?''', (value, goal_id))

                        cursor.execute('''UPDATE wellness_goals SET best_streak = current_streak
                            WHERE id = ? AND current_streak > best_streak''', (goal_id,))

                        # Award points using same connection
                        cursor.execute(
                            'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                            (20, user_id)
                        )
                    else:
                        cursor.execute('''UPDATE wellness_goals SET current_value = ?,
                            current_streak = 0 WHERE id = ?''', (value, goal_id))

                    conn.commit()

                    # Check for streak milestones
                    cursor.execute('SELECT current_streak, goal_type FROM wellness_goals WHERE id = ?',
                                  (goal_id,))
                    streak_result = cursor.fetchone()

                    if streak_result and is_completed:
                        streak, goal_type = streak_result
                        if streak in [7, 30, 90]:
                            bonus_points = {7: 100, 30: 500, 90: 1000}
                            bonus = bonus_points[streak]

                            # Award bonus points in same transaction
                            cursor.execute(
                                'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                                (bonus, user_id)
                            )
                            conn.commit()

                            return {"success": True, "message": f"Progress updated! {goal_type.capitalize()} Goal: {streak}-Day Streak! 🔥 +{bonus} points!"}

                    return {"success": True, "message": "Progress updated successfully."}
        except Exception as e:
            print(f"Error updating progress: {str(e)}")
            return {"success": False, "message": f"Error updating progress: {str(e)}"}


    def save_diet_plan(self, user_id: int, plan_data: Dict):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO diet_plans
                    (user_id, diet_preference, allergies, health_goal, calories_target, plan_data)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (user_id, plan_data.get('diet_preference'), plan_data.get('allergies'),
                     plan_data.get('health_goal'), plan_data.get('calories_target'),
                     json.dumps(plan_data.get('meals'))))
                conn.commit()
                return {"success": True}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_latest_diet_plan(self, user_id: int):
        try:
            with sqlite3.connect(self.db_name) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''SELECT plan_data, diet_preference, allergies, health_goal, calories_target
                    FROM diet_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1''', (user_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        'meals': json.loads(row['plan_data']),
                        'diet_preference': row['diet_preference'],
                        'allergies': row['allergies'],
                        'health_goal': row['health_goal'],
                        'calories_target': row['calories_target']
                    }
                return None
        except Exception as e:
            return None

    def create_community_post(self, user_id: int, title: str, content: str, anonymous: bool = False):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO community_posts (user_id, title, content, anonymous)
                    VALUES (?, ?, ?, ?)''', (user_id, title, content, 1 if anonymous else 0))
                conn.commit()
                return {"success": True, "post_id": cursor.lastrowid}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_community_posts(self, limit: int = 50) -> List[Dict]:
        """Get community posts with proper error handling"""
        try:
            with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''SELECT p.id, p.title, p.content, p.anonymous, p.likes,
                    p.created_at, u.name FROM community_posts p
                    LEFT JOIN users u ON p.user_id = u.id
                    ORDER BY p.created_at DESC LIMIT ?''', (limit,))

                posts = []
                for row in cursor.fetchall():
                    posts.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'anonymous': row['anonymous'],
                        'likes': row['likes'],
                        'date': row['created_at'],
                        'author': 'Anonymous' if row['anonymous'] else (row['name'] or 'Unknown')
                    })

                return posts
        except Exception as e:
            print(f"Error getting community posts: {str(e)}")
            return []



    def add_achievement(self, user_id: int, badge_name: str, badge_icon: str):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO achievements (user_id, badge_name, badge_icon) VALUES (?, ?, ?)',
                             (user_id, badge_name, badge_icon))
                conn.commit()
        except Exception as e:
            print(f"Error adding achievement: {str(e)}")

    def get_achievements(self, user_id: int) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_name) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT badge_name, badge_icon, earned_at FROM achievements WHERE user_id = ?',
                             (user_id,))
                achievements = [{'name': row['badge_name'], 'icon': row['badge_icon'], 'earned_at': row['earned_at']}
                              for row in cursor.fetchall()]
                return achievements
        except Exception as e:
            print(f"Error getting achievements: {str(e)}")
            return []

    def get_chat_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get user chat analytics with proper error handling"""
        try:
            with db_lock:
                with sqlite3.connect(self.db_name, check_same_thread=False, timeout=10) as conn:
                    cursor = conn.cursor()

                    # Total chats
                    cursor.execute('SELECT COUNT(*) FROM chat_history WHERE user_id = ?', (user_id,))
                    total_chats = cursor.fetchone()[0] or 0

                    # Sentiment distribution
                    cursor.execute('''SELECT sentiment, COUNT(*) FROM chat_history
                        WHERE user_id = ? AND sentiment IS NOT NULL GROUP BY sentiment''', (user_id,))
                    sentiment_data = dict(cursor.fetchall()) or {}

                    # Emotion distribution
                    cursor.execute('''SELECT emotion, COUNT(*) FROM chat_history
                        WHERE user_id = ? AND emotion IS NOT NULL GROUP BY emotion
                        ORDER BY COUNT(*) DESC''', (user_id,))
                    emotion_data = dict(cursor.fetchall()) or {}

                    # Activity data (last 7 days)
                    cursor.execute('''SELECT DATE(created_at) as date, COUNT(*)
                        FROM chat_history WHERE user_id = ? AND created_at >= date('now', '-7 days')
                        GROUP BY DATE(created_at) ORDER BY date''', (user_id,))
                    activity_data = dict(cursor.fetchall()) or {}

                    return {
                        'total_chats': total_chats,
                        'sentiment_distribution': sentiment_data,
                        'emotion_distribution': emotion_data,
                        'recent_activity': activity_data
                    }
        except Exception as e:
            print(f"Error getting analytics: {str(e)}")
            return {
                'total_chats': 0,
                'sentiment_distribution': {},
                'emotion_distribution': {},
                'recent_activity': {}
            }

    def get_user_count(self) -> int:
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                return cursor.fetchone()[0]
        except Exception:
            return 0

    # Add wellness points methods
    def award_points(self, user_id: int, points: int, reason: str, conn=None):
        """Award wellness points - accepts optional connection to avoid nesting"""
        try:
            should_close = False
            if conn is None:
                conn = sqlite3.connect(self.db_name, check_same_thread=False, timeout=10)
                should_close = True

            try:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                    (points, user_id)
                )
                conn.commit()
                print(f"✨ Awarded {points} points to user {user_id} for: {reason}")
            finally:
                if should_close:
                    conn.close()
        except Exception as e:
            print(f"Error awarding points: {e}")

    def get_user_points(self, user_id: int) -> int:
        """Get user's wellness points"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT wellness_points FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else 0

    def deduct_points(self, user_id: int, points: int) -> bool:
        """Deduct points from user (for store purchases)"""
        current_points = self.get_user_points(user_id)
        if current_points >= points:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET wellness_points = wellness_points - ? WHERE id = ?', (points, user_id))
                conn.commit()
            return True
        return False

    # User connections methods
    def send_buddy_request(self, user_id: int, buddy_email: str) -> Dict:
        """Send wellness buddy connection request"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE email = ?', (buddy_email,))
            buddy = cursor.fetchone()

            if not buddy:
                return {'success': False, 'message': 'User not found'}

            buddy_id = buddy[0]

            if buddy_id == user_id:
                return {'success': False, 'message': 'Cannot add yourself as buddy'}

            # Check if connection already exists
            cursor.execute('''SELECT id, status FROM user_connections
                WHERE (user_id = ? AND buddy_id = ?) OR (user_id = ? AND buddy_id = ?)''',
                (user_id, buddy_id, buddy_id, user_id))
            existing = cursor.fetchone()

            if existing:
                return {'success': False, 'message': 'Connection already exists'}

            cursor.execute('''INSERT INTO user_connections (user_id, buddy_id, status)
                VALUES (?, ?, 'pending')''', (user_id, buddy_id))
            conn.commit()

            return {'success': True, 'message': 'Buddy request sent!'}

    def accept_buddy_request(self, user_id: int, request_id: int) -> Dict:
        """Accept a buddy request"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''UPDATE user_connections
                SET status = 'accepted', accepted_at = ?
                WHERE id = ? AND buddy_id = ? AND status = 'pending' ''',
                (datetime.datetime.now(), request_id, user_id))

            if cursor.rowcount > 0:
                conn.commit()
                return {'success': True, 'message': 'Buddy request accepted!'}

            return {'success': False, 'message': 'Request not found'}

    def get_buddy_stats(self, user_id: int, buddy_id: int) -> Dict:
        """Get anonymized stats for a buddy"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()

            # Check if they're connected
            cursor.execute('''SELECT status FROM user_connections
                WHERE ((user_id = ? AND buddy_id = ?) OR (user_id = ? AND buddy_id = ?))
                AND status = 'accepted' ''',
                (user_id, buddy_id, buddy_id, user_id))

            if not cursor.fetchone():
                return {'success': False, 'message': 'Not connected'}

            # Get buddy's streak
            cursor.execute('''SELECT MAX(current_streak) FROM wellness_goals
                WHERE user_id = ? AND is_active = 1''', (buddy_id,))
            streak = cursor.fetchone()[0] or 0

            # Get recent activity count
            cursor.execute('''SELECT COUNT(*) FROM health_logs
                WHERE user_id = ? AND log_date >= date('now', '-7 days')''', (buddy_id,))
            activity_count = cursor.fetchone()[0]

            # Get goals completed today
            cursor.execute('''SELECT COUNT(*) FROM goal_progress gp
                JOIN wellness_goals wg ON gp.goal_id = wg.id
                WHERE wg.user_id = ? AND gp.date = date('now') AND gp.completed = 1''',
                (buddy_id,))
            goals_today = cursor.fetchone()[0]

            return {
                'success': True,
                'buddy_anonymous_name': f'Buddy_{buddy_id}',
                'current_streak': streak,
                'logs_this_week': activity_count,
                'goals_completed_today': goals_today
            }

    def get_unread_nudges(self, user_id: int) -> List[Dict]:
        """Get unread nudges for user"""
        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''SELECT id, type, priority, message, action, created_at
                FROM nudges WHERE user_id = ? AND read = 0
                ORDER BY priority DESC, created_at DESC LIMIT 5''', (user_id,))

            return [dict(row) for row in cursor.fetchall()]

    def mark_nudge_read(self, nudge_id: int):
        """Mark a nudge as read"""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE nudges SET read = 1 WHERE id = ?', (nudge_id,))
            conn.commit()

    def export_user_data(self, user_id: int) -> Dict:
        """Export all user data for download"""
        data = {}

        with sqlite3.connect(self.db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # User info
            cursor.execute('SELECT email, name, age, created_at FROM users WHERE id = ?', (user_id,))
            data['user_info'] = dict(cursor.fetchone())

            # Health logs
            cursor.execute('SELECT * FROM health_logs WHERE user_id = ? ORDER BY log_date DESC', (user_id,))
            data['health_logs'] = [dict(row) for row in cursor.fetchall()]

            # Journal entries
            cursor.execute('SELECT title, content, mood, sentiment, created_at FROM journal_entries WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
            data['journal_entries'] = [dict(row) for row in cursor.fetchall()]

            # Goals
            cursor.execute('SELECT * FROM wellness_goals WHERE user_id = ?', (user_id,))
            data['goals'] = [dict(row) for row in cursor.fetchall()]

            # Achievements
            cursor.execute('SELECT badge_name, points_awarded, earned_at FROM achievements WHERE user_id = ?', (user_id,))
            data['achievements'] = [dict(row) for row in cursor.fetchall()]

        return data

# ============================================================================
# ADVANCED EMOTION ANALYZER
# ============================================================================

class AdvancedEmotionAnalyzer:
    def __init__(self):
        self.stress_keywords = ['stressed', 'anxious', 'worried', 'pressure', 'overwhelmed', 'panic']
        self.sadness_keywords = ['sad', 'depressed', 'unhappy', 'lonely', 'down', 'blue', 'crying']
        self.anxiety_keywords = ['anxiety', 'nervous', 'panic', 'scared', 'afraid', 'fear']
        self.joy_keywords = ['happy', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'excited', 'joy']
        self.fatigue_keywords = ['tired', 'exhausted', 'drained', 'weary', 'sleepy']
        self.anger_keywords = ['angry', 'frustrated', 'annoyed', 'furious', 'mad', 'anger']
        self.crisis_keywords = ['suicide', 'self harm', 'harm myself', 'die', 'kill myself', 'end my life']

        # CBT thought patterns
        self.cbt_patterns = {
            'all_or_nothing': ['always', 'never', 'everything', 'nothing', 'completely', 'totally'],
            'overgeneralization': ['everyone', 'nobody', 'all the time', 'every time'],
            'catastrophizing': ['disaster', 'terrible', 'worst', 'horrible', 'awful', 'catastrophe'],
            'mind_reading': ['they think', 'he thinks', 'she thinks', 'probably thinks'],
            'should_statements': ['should', 'must', 'have to', 'ought to'],
            'emotional_reasoning': ['i feel like', 'i feel that', 'because i feel']
        }

    def analyze_sentiment(self, text: str) -> str:
        text_lower = text.lower()
        positive = ['happy', 'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'joy']
        negative = ['sad', 'bad', 'terrible', 'awful', 'pain', 'sick', 'horrible', 'stress', 'anxiety', 'anger', 'unhappy']

        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)

        return 'positive' if pos_count > neg_count else ('negative' if neg_count > pos_count else 'neutral')

    def detect_emotion(self, text: str) -> tuple:
        text_lower = text.lower()
        emotions = {
            'stress': sum(1 for word in self.stress_keywords if word in text_lower),
            'sadness': sum(1 for word in self.sadness_keywords if word in text_lower),
            'anxiety': sum(1 for word in self.anxiety_keywords if word in text_lower),
            'joy': sum(1 for word in self.joy_keywords if word in text_lower),
            'fatigue': sum(1 for word in self.fatigue_keywords if word in text_lower),
            'anger': sum(1 for word in self.anger_keywords if word in text_lower)
        }
        emotions['neutral'] = 1

        primary = max(emotions, key=emotions.get)
        score = min(max(emotions.values()) / 3.0, 10)
        return primary, score

    def detect_crisis(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)

    def detect_cbt_patterns(self, text: str) -> List[Dict]:
        """Detect negative thought patterns for CBT intervention"""
        text_lower = text.lower()
        detected_patterns = []

        for pattern_name, keywords in self.cbt_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_patterns.append({
                    'pattern': pattern_name,
                    'reframe_prompt': self._get_cbt_reframe_prompt(pattern_name),
                    'example': self._get_cbt_example(pattern_name)
                })

        return detected_patterns

    def _get_cbt_reframe_prompt(self, pattern: str) -> str:
        """Get CBT reframing prompts"""
        prompts = {
            'all_or_nothing': "Is this really all or nothing? What's a more balanced way to see this?",
            'overgeneralization': "Is this always true, or are there exceptions? What evidence supports this?",
            'catastrophizing': "What's the worst that could realistically happen? What's more likely to occur?",
            'mind_reading': "Do I really know what they're thinking? What else could they be thinking?",
            'should_statements': "Is this 'should' helping or adding pressure? What would be more flexible?",
            'emotional_reasoning': "Just because I feel this way, does that make it fact? What do the facts say?"
        }
        return prompts.get(pattern, "What's another way to look at this thought?")

    def _get_cbt_example(self, pattern: str) -> str:
        """Get CBT reframing examples"""
        examples = {
            'all_or_nothing': "Instead of 'I always fail,' try 'Sometimes I succeed, sometimes I don't—that's human.'",
            'overgeneralization': "Instead of 'Nobody likes me,' try 'Some people like me, and I'm working on new connections.'",
            'catastrophizing': "Instead of 'This is a disaster,' try 'This is challenging, but I can handle it.'",
            'mind_reading': "Instead of 'They think I'm stupid,' try 'I don't know what they think. I can ask if needed.'",
            'should_statements': "Instead of 'I should be perfect,' try 'I'm doing my best, and that's enough.'",
            'emotional_reasoning': "Instead of 'I feel like a failure,' try 'I feel disappointed, but feelings aren't facts.'"
        }
        return examples.get(pattern, "Reframe with evidence and balance.")

    def get_empathetic_response(self, emotion: str) -> str:
        responses = {
            'stress': "I understand you're stressed. Try deep breathing: 4 counts in, hold 4, out 4. Take small breaks.",
            'sadness': "I'm sorry you're feeling down. It's normal to feel this way. Would wellness tips help?",
            'anxiety': "Anxiety is manageable. Try grounding: name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
            'joy': "That's wonderful! Keep this positive energy and maintain your wellness habits!",
            'fatigue': "Rest is important. Get 7-8 hours sleep, stay hydrated, and move gently.",
            'anger': "Take a moment to breathe. Your feelings are valid. Would you like some calming techniques?",
            'neutral': "Thanks for sharing your thoughts. I'm here if you have any wellness questions."
        }
        return responses.get(emotion, "I'm here to support you.")


# ============================================================================
# HEALTH KNOWLEDGE BASE
# ============================================================================

class HealthKnowledgeBase:
    def __init__(self):
        self.kb = {
            "symptoms": {
                "headache": "Drink water, rest in quiet room, avoid screens. If persists >24h, see doctor.",
                "fever": "Stay hydrated, rest, monitor temperature. Seek help if >102°F or lasts >3 days.",
                "cough": "Stay hydrated, use honey for adults. See doctor if lasts >3 weeks.",
                "stomach pain": "Avoid heavy meals, stay hydrated, rest. Urgent if severe with vomiting.",
                "sore throat": "Gargle warm salt water, stay hydrated. See doctor if >1 week.",
                "dizziness": "Sit/lie down, drink water. Emergency if with chest pain or unconsciousness.",
                "nausea": "Eat bland foods, stay hydrated with clear fluids. See doctor if persistent.",
                "back pain": "Ice first 48h, then heat. See doctor if radiating down legs.",
                "chest pain": "⚠️ EMERGENCY: If severe with shortness of breath, call 911 immediately!",
                "cold": "Rest, fluids, saline drops. See doctor if symptoms worsen.",
                "flu": "Rest, fluids, fever reducers. Stay isolated. See doctor if breathing difficulty.",
                "allergy": "Avoid triggers, take antihistamines. See doctor if persistent.",
                "insomnia": "Regular schedule, no caffeine late, relaxing routine. Consult if chronic.",
                "anxiety": "Deep breathing, exercise, limit caffeine. Seek professional help if severe.",
                "fatigue": "Ensure sleep, balanced diet, exercise. See doctor if persistent.",
                "joint pain": "Rest, ice, elevation. Gentle stretching may help.",
                "rash": "Keep clean/dry, don't scratch, moisturize. See doctor if spreading.",
                "constipation": "Increase fiber, drink water, exercise. See doctor if >3 days.",
                "diarrhea": "Stay hydrated, eat bland foods. Seek care if severe or bloody.",
                "vomiting": "Sip clear fluids, rest. Get help if persistent with severe pain."
            }
        }

    def get_response(self, intent, entity=None):
        if entity and entity in self.kb.get("symptoms", {}):
            return self.kb["symptoms"][entity]
        return "I'm here to help with health questions. Please describe your symptoms or ask about wellness."

# ============================================================================
# MULTILINGUAL TRANSLATOR
# ============================================================================

class MultilingualTranslator:
    LANG_CODE_MAP = {
        'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
        'de': 'German', 'it': 'Italian', 'pt': 'Portuguese'
    }

    def __init__(self):
        self.translators = {}

    def detect_language(self, text: str) -> str:
        try:
            if not text.strip():
                return 'en'
            return detect(text.strip())
        except Exception:
            return 'en'

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            if source_lang == target_lang:
                return text
            key = f"{source_lang}_{target_lang}"
            if key not in self.translators:
                self.translators[key] = GoogleTranslator(source=source_lang, target=target_lang)
            return self.translators[key].translate(text) or text
        except Exception:
            return text

# ============================================================================
# HEALTH CHATBOT INTEGRATION LAYER
# ============================================================================



class HealthChatbot:
    """Integration layer that combines all components for intelligent responses"""

    def __init__(self, emotion_analyzer, translator, health_kb):
        self.emotion_analyzer = emotion_analyzer
        self.translator = translator
        self.health_kb = health_kb

        # Intent keywords mapping
        self.intent_keywords = {
            'diet': ['diet', 'food', 'eat', 'meal', 'nutrition', 'calorie'],
            'fitness': ['exercise', 'workout', 'fitness', 'gym', 'training'],
            'sleep': ['sleep', 'insomnia', 'rest', 'tired', 'fatigue'],
            'stress': ['stress', 'anxiety', 'worried', 'nervous', 'panic'],
            'hydration': ['water', 'drink', 'hydration', 'hydrate'],
            'weight': ['weight', 'lose', 'gain', 'fat', 'slim'],
            'greeting': ['hello', 'hi', 'hey', 'greetings']
        }

    def _detect_intent(self, text: str) -> tuple:
        """Detect user intent and extract symptom entity"""
        text_lower = text.lower()

        # Check for symptoms first
        for symptom in self.health_kb.kb["symptoms"].keys():
            if symptom in text_lower:
                return ('symptom', symptom)

        # Check for other intents
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return (intent, None)

        return ('general', None)

    def get_response(self, user_message: str, user_id: int = None) -> dict:
        """
        Main method to get chatbot response
        Returns dict with: response, detected_language, sentiment, emotion, emotion_score
        """
        # Step 1: Language detection and translation
        detected_lang = self.translator.detect_language(user_message)
        english_message = user_message
        if detected_lang != 'en':
            english_message = self.translator.translate_text(user_message, detected_lang, 'en')

        # Step 2: Emotion and sentiment analysis
        emotion, emotion_score = self.emotion_analyzer.detect_emotion(english_message)
        sentiment = self.emotion_analyzer.analyze_sentiment(english_message)

        # Step 3: Crisis detection
        if self.emotion_analyzer.detect_crisis(english_message):
            crisis_response = ("🚨 I detect you might be in crisis. Please reach out:\n"
                             "• Call 911 or your local emergency number\n"
                             "• Suicide Prevention Lifeline: 988 (US/Canada)\n"
                             "• Crisis Text Line: Text HOME to 741741")

            # Translate crisis response back to user's language
            if detected_lang != 'en':
                crisis_response = self.translator.translate_text(crisis_response, 'en', detected_lang)

            return {
                'response': crisis_response,
                'detected_language': detected_lang,
                'sentiment': 'crisis',
                'emotion': 'crisis',
                'emotion_score': 10.0
            }

        # Step 4: Intent detection
        intent, entity = self._detect_intent(english_message)

        # Step 5: Generate response based on intent (in English)
        english_response = self._generate_response(intent, entity, emotion)

        # Step 6: Translate response back to user's language
        final_response = english_response
        if detected_lang != 'en':
            final_response = self.translator.translate_text(english_response, 'en', detected_lang)

        return {
            'response': final_response,
            'detected_language': detected_lang,
            'sentiment': sentiment,
            'emotion': emotion,
            'emotion_score': emotion_score
        }

    def _generate_response(self, intent: str, entity: str, emotion: str) -> str:
        """Generate response based on intent and emotion"""

        # Get empathetic response if needed
        empathetic = ""
        if emotion not in ['neutral', 'joy']:
            empathetic = self.emotion_analyzer.get_empathetic_response(emotion) + "\n\n"

        # Generate main response based on intent
        if intent == 'symptom' and entity:
            main_response = self.health_kb.get_response('symptom', entity)
            return f"{empathetic}I understand you're concerned about {entity}. {main_response}"

        elif intent == 'diet':
            return (f"{empathetic}I can help you with nutrition! For a personalized diet plan, "
                   "please use the 'Diet Plan' feature. In general:\n"
                   "• Eat a balanced diet with vegetables, proteins, and whole grains\n"
                   "• Stay hydrated (8 glasses of water daily)\n"
                   "• Limit processed foods and sugar\n"
                   "• Eat regular meals")

        elif intent == 'fitness':
            return (f"{empathetic}Great that you're interested in fitness! For a personalized "
                   "workout plan, check the 'Fitness Plan' feature. General tips:\n"
                   "• Aim for 150 minutes of moderate exercise weekly\n"
                   "• Mix cardio and strength training\n"
                   "• Always warm up and cool down\n"
                   "• Listen to your body and rest when needed")

        elif intent == 'sleep':
            return (f"{empathetic}Sleep is crucial for health! Here are some tips:\n"
                   "• Maintain a regular sleep schedule\n"
                   "• Create a relaxing bedtime routine\n"
                   "• Avoid screens 1 hour before bed\n"
                   "• Keep your bedroom cool and dark\n"
                   "• Limit caffeine after 2 PM")

        elif intent == 'stress':
            return (f"{empathetic}I hear that you're feeling stressed. Here are immediate techniques:\n"
                   "• Deep breathing: Breathe in for 4, hold for 4, out for 4\n"
                   "• Progressive muscle relaxation\n"
                   "• Take a short walk\n"
                   "• Try our meditation exercises\n"
                   "• Talk to someone you trust")

        elif intent == 'hydration':
            return (f"{empathetic}Staying hydrated is essential! Here's how:\n"
                   "• Drink 8-10 glasses of water daily\n"
                   "• Drink more during exercise\n"
                   "• Carry a water bottle\n"
                   "• Eat water-rich foods (cucumbers, watermelon)\n"
                   "• Track your water intake in the Health Logger")

        elif intent == 'weight':
            return (f"{empathetic}I can help with weight management! Key principles:\n"
                   "• Sustainable changes work better than crash diets\n"
                   "• Balance calories in vs calories out\n"
                   "• Exercise regularly\n"
                   "• Get adequate sleep and manage stress\n"
                   "• Use our Diet and Fitness Plan features for personalized guidance!")

        elif intent == 'greeting':
            return ("Hello! 👋 I'm your AI wellness companion. I can help you with:\n"
                   "• Symptom guidance and health advice\n"
                   "• Diet and nutrition tips\n"
                   "• Fitness and exercise guidance\n"
                   "• Stress management and meditation\n"
                   "• Tracking your health journey\n\n"
                   "How can I assist you today?")

        else:  # general
            return (f"{empathetic}I'm here to help with your wellness! You can ask me about:\n"
                   "• Symptoms and health concerns\n"
                   "• Diet and nutrition\n"
                   "• Fitness and exercise\n"
                   "• Mental health and stress\n"
                   "• Sleep and hydration\n\n"
                   "What would you like to know?")

# ============================================================================
# DIET & FITNESS PLAN GENERATOR
# ============================================================================

class DietFitnessPlanGenerator:
    def generate_diet_plan(self, preference: str, allergies: str, goal: str, calories: int):
        meal_plans = {
            'vegetarian': {
                'weight_loss': {
                    'breakfast': ['Oatmeal with berries (300 cal)', 'Greek yogurt with nuts (250 cal)'],
                    'lunch': ['Quinoa salad with chickpeas (400 cal)', 'Vegetable stir-fry with tofu (380 cal)'],
                    'dinner': ['Grilled vegetables with brown rice (350 cal)', 'Chickpea curry (400 cal)'],
                    'snacks': ['Apple with almond butter (150 cal)', 'Carrot sticks with hummus (100 cal)']
                },
                'muscle_gain': {
                    'breakfast': ['Protein smoothie with banana (450 cal)', 'Eggs with avocado toast (500 cal)'],
                    'lunch': ['Paneer tikka with rice (550 cal)', 'Bean burrito bowl (600 cal)'],
                    'dinner': ['Lentil dal with rice (500 cal)', 'Veggie protein bowl (550 cal)'],
                    'snacks': ['Protein bar (200 cal)', 'Cottage cheese with fruit (180 cal)']
                },
                'maintenance': {
                    'breakfast': ['Whole grain toast with peanut butter (350 cal)', 'Vegetable omelet (320 cal)'],
                    'lunch': ['Buddha bowl (450 cal)', 'Veggie wrap (420 cal)'],
                    'dinner': ['Vegetable pasta (450 cal)', 'Stir-fried tofu (430 cal)'],
                    'snacks': ['Fresh fruit (80 cal)', 'Trail mix (150 cal)']
                }
            },
            'non-veg': {
                'weight_loss': {
                    'breakfast': ['Egg white omelet (200 cal)', 'Turkey sausage with toast (280 cal)'],
                    'lunch': ['Grilled chicken salad (350 cal)', 'Tuna wrap (320 cal)'],
                    'dinner': ['Grilled fish with vegetables (300 cal)', 'Chicken breast with quinoa (350 cal)'],
                    'snacks': ['Boiled eggs (140 cal)', 'Jerky (100 cal)']
                },
                'muscle_gain': {
                    'breakfast': ['Scrambled eggs with bacon (450 cal)', 'Chicken breakfast burrito (500 cal)'],
                    'lunch': ['Grilled chicken with rice (600 cal)', 'Salmon with sweet potato (580 cal)'],
                    'dinner': ['Grilled steak with vegetables (550 cal)', 'Baked fish with quinoa (500 cal)'],
                    'snacks': ['Protein shake (250 cal)', 'Tuna can (200 cal)']
                },
                'maintenance': {
                    'breakfast': ['Eggs with toast (350 cal)', 'Chicken sausage with fruit (320 cal)'],
                    'lunch': ['Chicken sandwich (420 cal)', 'Fish tacos (450 cal)'],
                    'dinner': ['Grilled chicken with rice (450 cal)', 'Baked salmon (430 cal)'],
                    'snacks': ['Cottage cheese (100 cal)', 'Protein bar (180 cal)']
                }
            },
            'vegan': {
                'weight_loss': {
                    'breakfast': ['Chia pudding (250 cal)', 'Avocado toast (280 cal)'],
                    'lunch': ['Quinoa Buddha bowl (380 cal)', 'Lentil salad (350 cal)'],
                    'dinner': ['Tofu stir-fry (340 cal)', 'Chickpea curry (360 cal)'],
                    'snacks': ['Hummus with veggies (100 cal)', 'Fruit bowl (80 cal)']
                },
                'muscle_gain': {
                    'breakfast': ['Tofu scramble (400 cal)', 'Protein oatmeal (450 cal)'],
                    'lunch': ['Lentil dal with rice (550 cal)', 'Quinoa protein bowl (580 cal)'],
                    'dinner': ['Seitan stir-fry (520 cal)', 'Chickpea pasta (550 cal)'],
                    'snacks': ['Protein shake (240 cal)', 'Nut butter on rice cakes (200 cal)']
                },
                'maintenance': {
                    'breakfast': ['Overnight oats (350 cal)', 'Smoothie bowl (380 cal)'],
                    'lunch': ['Veggie wrap (420 cal)', 'Lentil soup (400 cal)'],
                    'dinner': ['Vegetable curry (430 cal)', 'Pasta primavera (450 cal)'],
                    'snacks': ['Fresh fruit (80 cal)', 'Trail mix (150 cal)']
                }
            }
        }

        plan = meal_plans.get(preference, meal_plans['vegetarian']).get(goal, meal_plans[preference]['maintenance'])

        return {
            'diet_preference': preference,
            'health_goal': goal,
            'calories_target': calories,
            'allergies': allergies,
            'meals': plan,
            'tips': [
                'Drink at least 8 glasses of water daily',
                'Eat slowly and mindfully',
                'Include colorful vegetables in every meal',
                'Prep meals in advance for better adherence',
                'Listen to your body\'s hunger cues'
            ]
        }

    def generate_fitness_plan(self, fitness_level: str, goal: str):
        plans = {
            'beginner': {
                'weight_loss': {
                    'monday': '20 min brisk walk + 10 min stretching',
                    'tuesday': '15 min bodyweight exercises (squats, lunges, push-ups)',
                    'wednesday': 'Rest or light yoga',
                    'thursday': '20 min cycling or swimming',
                    'friday': '15 min HIIT workout',
                    'saturday': '30 min outdoor activity',
                    'sunday': 'Active rest - walking or gentle yoga'
                },
                'muscle_gain': {
                    'monday': 'Upper body: Push-ups, dumbbell press (3x10)',
                    'tuesday': 'Lower body: Squats, lunges (3x12)',
                    'wednesday': 'Rest day',
                    'thursday': 'Core: Planks, crunches, leg raises (3x15)',
                    'friday': 'Full body circuit (3 rounds)',
                    'saturday': 'Cardio: 20 min light jog',
                    'sunday': 'Rest or stretching'
                },
                'maintenance': {
                    'monday': '30 min moderate cardio',
                    'tuesday': '20 min strength training',
                    'wednesday': '30 min yoga or pilates',
                    'thursday': '20 min HIIT',
                    'friday': '30 min outdoor activity',
                    'saturday': 'Full body workout',
                    'sunday': 'Rest or light stretching'
                }
            },
            'intermediate': {
                'weight_loss': {
                    'monday': '30 min running + core workout',
                    'tuesday': 'HIIT circuit training (25 min)',
                    'wednesday': '45 min cycling',
                    'thursday': 'Strength training full body',
                    'friday': '30 min swimming or rowing',
                    'saturday': 'Long cardio session (45-60 min)',
                    'sunday': 'Active recovery - yoga'
                },
                'muscle_gain': {
                    'monday': 'Chest & triceps (heavy weights, 4x8-10)',
                    'tuesday': 'Back & biceps (4x8-10)',
                    'wednesday': 'Legs (squats, deadlifts, 4x8-10)',
                    'thursday': 'Shoulders & abs',
                    'friday': 'Upper body hypertrophy (4x12-15)',
                    'saturday': 'Lower body hypertrophy',
                    'sunday': 'Rest or light cardio'
                },
                'maintenance': {
                    'monday': '40 min cardio',
                    'tuesday': 'Upper body strength',
                    'wednesday': 'Yoga or pilates',
                    'thursday': 'Lower body strength',
                    'friday': 'HIIT or circuit training',
                    'saturday': 'Sports or outdoor activity',
                    'sunday': 'Rest day'
                }
            },
            'advanced': {
                'weight_loss': {
                    'monday': '45 min HIIT + abs',
                    'tuesday': 'Strength & cardio combo',
                    'wednesday': '60 min intense cardio',
                    'thursday': 'Full body circuit training',
                    'friday': 'Tabata intervals + core',
                    'saturday': 'Long endurance training',
                    'sunday': 'Active recovery'
                },
                'muscle_gain': {
                    'monday': 'Chest focus (5 exercises, 4-5 sets)',
                    'tuesday': 'Back focus (5 exercises, 4-5 sets)',
                    'wednesday': 'Legs (heavy compound lifts)',
                    'thursday': 'Shoulders & arms',
                    'friday': 'Upper body power day',
                    'saturday': 'Lower body power day',
                    'sunday': 'Active rest or light cardio'
                },
                'maintenance': {
                    'monday': 'Push workout',
                    'tuesday': 'Pull workout',
                    'wednesday': 'Legs',
                    'thursday': 'Cardio & core',
                    'friday': 'Full body strength',
                    'saturday': 'Sports or activity',
                    'sunday': 'Rest or yoga'
                }
            }
        }

        return plans.get(fitness_level, plans['beginner']).get(goal, plans[fitness_level]['maintenance'])

# ============================================================================
# MEDITATION GUIDE
# ============================================================================

class MeditationGuide:
    MEDITATIONS = {
        "breathing": "🧘 5-Minute Breathing Exercise:\n1. Find a quiet place\n2. Breathe in for 4 counts\n3. Hold for 4 counts\n4. Exhale for 4 counts\n5. Repeat 10 times\nRelax and feel your stress melt away.",
        "body_scan": "🧘 Body Scan Meditation:\n1. Lie down comfortably\n2. Close your eyes\n3. Starting from toes, notice each body part\n4. Relax each area as you notice it\n5. Move upward through your body\n\nTakes about 10 minutes. Perfect before sleep.",
        "gratitude": "🙏 Gratitude Practice:\n1. Close your eyes\n2. Think of 3 things you're grateful for\n3. Feel the emotion attached to each\n4. Sit with this feeling for 2 minutes\n\nDo this daily for a positive mindset shift.",
        "stress_relief": "😌 Quick Stress Relief:\n1. Tense all muscles for 5 seconds\n2. Release suddenly and relax\n3. Take 3 deep breaths\n4. Repeat 5 times\n\nGreat for quick anxiety relief!"
    }

    @staticmethod
    def get_meditation(meditation_type: str) -> str:
        return MeditationGuide.MEDITATIONS.get(meditation_type, "Try our breathing or body scan meditation!")

# ============================================================================
# WELLNESS QUOTE GENERATOR
# ============================================================================

class WellnessQuoteGenerator:
    quotes = [
        "Take care of your body. It's the only place you have to live.",
        "Health is not about the weight you lose, but the life you gain.",
        "Your body hears everything your mind says. Stay positive.",
        "The groundwork of all happiness is health.",
        "Invest in yourself. Your body, your mind, your spirit.",
        "Small daily improvements are the key to long-term results.",
        "Wellness is the complete integration of body, mind, and spirit.",
        "Progress, not perfection.",
        "Believe in yourself and you will be unstoppable.",
        "The only bad workout is the one you didn't do.",
        "Make yourself a priority once in a while.",
        "You don't have to be great to start, but you have to start to be great."
    ]

    @staticmethod
    def get_random_quote():
        return random.choice(WellnessQuoteGenerator.quotes)

# ============================================================================
# DAILY CHALLENGES
# ============================================================================

DAILY_CHALLENGES = [
    {"name": "10K Steps", "description": "Walk 10,000 steps today", "difficulty": "medium", "reward": 50},
    {"name": "Water Challenge", "description": "Drink 8 glasses of water", "difficulty": "easy", "reward": 30},
    {"name": "Meditation", "description": "Meditate for 10 minutes", "difficulty": "easy", "reward": 40},
    {"name": "No Sugar Day", "description": "Avoid all sugary foods", "difficulty": "hard", "reward": 100},
    {"name": "Early Sleep", "description": "Sleep by 10 PM", "difficulty": "medium", "reward": 50},
    {"name": "Veggie Power", "description": "Eat 5 servings of vegetables", "difficulty": "medium", "reward": 60},
    {"name": "Cardio Blast", "description": "30 minutes of cardio exercise", "difficulty": "hard", "reward": 80},
    {"name": "Gratitude Journal", "description": "Write 5 things you're grateful for", "difficulty": "easy", "reward": 35}
]

def setup_sqlite():
    """Setup SQLite for better concurrent access"""
    try:
        with sqlite3.connect(DATABASE_NAME, check_same_thread=False, timeout=10) as conn:
            cursor = conn.cursor()
            # Enable WAL mode for better concurrency
            cursor.execute('PRAGMA journal_mode=WAL')
            # Increase busy timeout
            cursor.execute('PRAGMA busy_timeout=10000')
            conn.commit()
    except Exception as e:
        print(f"SQLite setup warning: {e}")

# ============================================================================
# INITIALIZE MANAGERS
# ============================================================================

setup_sqlite()

db_manager = DatabaseManager()
emotion_analyzer = AdvancedEmotionAnalyzer()
diet_fitness_gen = DietFitnessPlanGenerator()
quote_generator = WellnessQuoteGenerator()
translator = MultilingualTranslator()
health_kb = HealthKnowledgeBase()
meditation_guide = MeditationGuide()
health_chatbot = HealthChatbot(emotion_analyzer, translator, health_kb)
llm_service = LLMService(provider="mock")
db_manager = DatabaseManager()
emotion_analyzer = AdvancedEmotionAnalyzer()
micro_habit_stacker = MicroHabitStacker(db_manager)

# ============================================================================
# FASTAPI APP
# ============================================================================

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Global Wellness Platform v7.0",
    description="AI-powered wellness with LLM integration, CBT, and social features",
    version="7.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])



health_chatbot = None  # Will initialize after defining HealthChatbot class
nudge_scheduler = None  # Will initialize after server starts
# Add a simple route to verify the app is working
@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup"""
    global health_chatbot, nudge_scheduler

    # Create health chatbot instance
    health_chatbot = HealthChatbot(emotion_analyzer, translator, health_kb)


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def get_frontend():
    return HTMLResponse(content=HTML_FRONTEND)

@app.post("/api/register")
async def register_user(user_data: UserRegistration):
    try:
        result = db_manager.create_user(
            email=user_data.email, password=user_data.password, name=user_data.name,
            age=user_data.age, language_preference=user_data.languagePreference
        )
        if result["success"]:
            return JSONResponse(status_code=201, content={"message": "Registration successful!"})
        else:
            raise HTTPException(status_code=400, detail=result['message'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login_user(login_data: UserLogin):
    try:
        user_data = db_manager.authenticate_user(login_data.email, login_data.password)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = jwt.encode({
            'user_id': user_data['id'],
            'email': user_data['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS)
        }, user_data['secret_key'], algorithm='HS256')

        return {"message": f"Welcome back, {user_data['name']}!", "token": token, "user": user_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_user_from_token(authorization: Optional[str] = Header(None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        token = authorization.split(" ")[1]
        import base64
        payload_part = token.split('.')[1]
        payload_part += '=' * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part))
        user_id = payload.get('user_id')
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        with sqlite3.connect(db_manager.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT secret_key FROM users WHERE id = ?', (user_id,))
            secret_key_row = cursor.fetchone()
            if not secret_key_row:
                raise HTTPException(status_code=401, detail="User not found")
            user_secret_key = secret_key_row[0]

        jwt.decode(token, user_secret_key, algorithms=['HS256'])
        return user_id
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.post("/api/chat")
async def chat_with_bot(request: dict, authorization: Optional[str] = Header(None)):
    try:
        user_message = request.get("message", "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        user_id = None
        if authorization and authorization.startswith("Bearer "):
            try:
                user_id = get_user_from_token(authorization)
            except:
                pass

        # Use the integrated chatbot to get response
        response_data = health_chatbot.get_response(user_message, user_id)

        # Save to database if user is logged in
        if user_id:
            db_manager.save_chat_message(
                user_id=user_id,
                user_message=user_message,
                bot_response=response_data['response'],
                detected_language=response_data['detected_language'],
                sentiment=response_data['sentiment'],
                emotion=response_data['emotion'],
                emotion_score=response_data['emotion_score']
            )

        return {
            "user_message": user_message,
            "bot_response": response_data['response'],
            "detected_language": response_data['detected_language'],
            "sentiment": response_data['sentiment'],
            "emotion": response_data['emotion'],
            "timestamp": datetime.datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    try:
        return {
            "totalUsers": db_manager.get_user_count(),
            "databaseStatus": "Connected",
            "serverStatus": "Running",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/health-log")
async def save_health_log(log_data: HealthLog, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    log_dict = {k: v for k, v in log_data.dict().items() if v is not None}
    result = db_manager.save_health_log(user_id, log_dict)
    if result["success"]:
        return JSONResponse(status_code=200, content=result)
    else:
        raise HTTPException(status_code=400, detail=result["message"])

@app.get("/api/health-logs")
async def get_health_logs(authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    return db_manager.get_health_logs(user_id, days=30)

@app.post("/api/journal")
async def save_journal(entry: JournalEntry, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    sentiment = emotion_analyzer.analyze_sentiment(entry.content)
    emotion, emotion_score = emotion_analyzer.detect_emotion(entry.content)
    result = db_manager.save_journal_entry(user_id, entry.title, entry.content,
                                          mood=emotion, sentiment=sentiment, emotion_score=emotion_score)
    return result

@app.get("/api/journal")
async def get_journal(authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    return db_manager.get_journal_entries(user_id)

@app.post("/api/goals")
async def create_goal(goal: GoalCreate, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    result = db_manager.save_goal(user_id, goal.goal_type, goal.target_value, goal.description)
    return result

@app.get("/api/goals")
async def get_goals(authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    return db_manager.get_user_goals(user_id)

@app.post("/api/goals/{goal_id}/progress")
async def update_goal(goal_id: int, request: dict, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    value = request.get("value", 0)
    result = db_manager.update_goal_progress(goal_id, value)
    return result

@app.post("/api/diet-plan")
async def create_diet_plan(plan_request: DietPlanRequest, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)

    plan = diet_fitness_gen.generate_diet_plan(
        plan_request.diet_preference,
        plan_request.allergies,
        plan_request.health_goal,
        plan_request.calories_target
    )

    db_manager.save_diet_plan(user_id, {
        **plan_request.dict(),
        'meals': plan['meals']
    })

    return plan

@app.get("/api/diet-plan")
async def get_diet_plan(authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    plan = db_manager.get_latest_diet_plan(user_id)

    if plan:
        full_plan = diet_fitness_gen.generate_diet_plan(
            plan.get('diet_preference', 'vegetarian'),
            plan.get('allergies'),
            plan.get('health_goal', 'maintenance'),
            plan.get('calories_target', 2000)
        )
        full_plan['meals'] = plan['meals']
        return full_plan

    return {"message": "No diet plan found"}

@app.post("/api/fitness-plan")
async def create_fitness_plan(request: dict, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    fitness_level = request.get("fitness_level", "beginner")
    goal = request.get("goal", "maintenance")

    plan = diet_fitness_gen.generate_fitness_plan(fitness_level, goal)
    return {"fitness_level": fitness_level, "goal": goal, "weekly_plan": plan}

@app.post("/api/community/posts")
async def create_post(post: CommunityPost, authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    result = db_manager.create_community_post(user_id, post.title, post.content, post.anonymous)
    return result

@app.get("/api/community/posts")
async def get_posts():
    return db_manager.get_community_posts(limit=50)

@app.get("/api/meditation/{meditation_type}")
async def get_meditation(meditation_type: str):
    return {"meditation": meditation_guide.get_meditation(meditation_type)}

@app.get("/api/challenges")
async def get_challenges():
    return DAILY_CHALLENGES

@app.get("/api/achievements")
async def get_achievements(authorization: Optional[str] = Header(None)):
    user_id = get_user_from_token(authorization)
    return db_manager.get_achievements(user_id)

@app.post("/api/complete-challenge")
async def complete_challenge(challenge_data: dict, authorization: Optional[str] = Header(None)):
    """Complete a challenge"""
    user_id = get_user_from_token(authorization)
    challenge_name = challenge_data.get("challenge_name")

    try:
        with db_lock:
            with sqlite3.connect(db_manager.db_name, check_same_thread=False, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO achievements (user_id, badge_name, badge_icon) VALUES (?, ?, ?)',
                             (user_id, challenge_name, "⭐"))

                # Award points in same transaction
                cursor.execute(
                    'UPDATE users SET wellness_points = wellness_points + ? WHERE id = ?',
                    (50, user_id)
                )

                conn.commit()

        return {"success": True, "message": f"Congratulations! You completed {challenge_name}!"}
    except Exception as e:
        print(f"Error completing challenge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quote")
async def get_daily_quote():
    return {"quote": quote_generator.get_random_quote()}

@app.get("/api/analytics")
@app.get("/api/analytics")
@app.get("/api/analytics")
async def get_analytics(authorization: Optional[str] = Header(None)):
    """Get user analytics for dashboard"""
    try:
        user_id = get_user_from_token(authorization)

        # Get analytics
        analytics = db_manager.get_chat_analytics(user_id)

        # Get health logs
        logs = db_manager.get_health_logs(user_id, days=30)

        # Get goals
        goals = db_manager.get_user_goals(user_id)

        # Calculate total streak from all goals
        total_streak = 0
        for goal in goals:
            current_streak = goal.get('current_streak', 0) if isinstance(goal, dict) else getattr(goal, 'current_streak', 0)
            if current_streak:
                total_streak += current_streak

        return {
            'total_chats': analytics.get('total_chats', 0),
            'health_logs_count': len(logs) if logs else 0,
            'sentiment_distribution': analytics.get('sentiment_distribution', {}),
            'emotion_distribution': analytics.get('emotion_distribution', {}),
            'recent_activity': analytics.get('recent_activity', {}),
            'active_goals': len(goals) if goals else 0,
            'total_streak': total_streak
        }
    except Exception as e:
        print(f"Error in get_analytics: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return safe defaults instead of failing
        return {
            'total_chats': 0,
            'health_logs_count': 0,
            'sentiment_distribution': {},
            'emotion_distribution': {},
            'recent_activity': {},
            'active_goals': 0,
            'total_streak': 0
        }
@app.post("/api/llm/diet-plan")
async def generate_llm_diet_plan(request: dict, authorization: Optional[str] = Header(None)):
    """Generate AI-powered personalized diet plan"""
    user_id = get_user_from_token(authorization)

    user_profile = {
        'diet_preference': request.get('diet_preference', 'vegetarian'),
        'health_goal': request.get('health_goal', 'maintenance'),
        'allergies': request.get('allergies', ''),
        'calories_target': request.get('calories_target', 2000)
    }

    # Get user's health logs for personalization
    health_logs = db_manager.get_health_logs(user_id, days=30)

    # Generate plan using LLM
    plan = llm_service.generate_diet_plan(user_profile, health_logs)

    # Save to database
    db_manager.save_diet_plan(user_id, {**user_profile, 'meals': plan['meals']})

    return {
        **plan,
        'message': 'Personalized diet plan generated using AI!',
        'personalization_note': f'Based on your last {len(health_logs)} days of health data'
    }

@app.post("/api/llm/fitness-plan")
async def generate_llm_fitness_plan(request: dict, authorization: Optional[str] = Header(None)):
    """Generate AI-powered personalized fitness plan"""
    user_id = get_user_from_token(authorization)

    user_profile = {
        'fitness_level': request.get('fitness_level', 'beginner'),
        'goal': request.get('goal', 'maintenance')
    }

    health_logs = db_manager.get_health_logs(user_id, days=30)

    plan = llm_service.generate_fitness_plan(user_profile, health_logs)

    return plan

@app.get("/api/nudges")
async def get_nudges(authorization: Optional[str] = Header(None)):
    """Get proactive wellness nudges"""
    user_id = get_user_from_token(authorization)

    nudges = db_manager.get_unread_nudges(user_id)

    return {
        'nudges': nudges,
        'count': len(nudges)
    }

@app.post("/api/nudges/{nudge_id}/read")
async def mark_nudge_read(nudge_id: int, authorization: Optional[str] = Header(None)):
    """Mark a nudge as read"""
    user_id = get_user_from_token(authorization)
    db_manager.mark_nudge_read(nudge_id)

    return {'success': True}

@app.get("/api/points")
async def get_wellness_points(authorization: Optional[str] = Header(None)):
    """Get user's wellness points balance"""
    user_id = get_user_from_token(authorization)
    points = db_manager.get_user_points(user_id)

    return {
        'points': points,
        'rank': 'Wellness Warrior' if points > 1000 else 'Wellness Explorer'
    }

@app.get("/api/store")
async def get_store_items():
    """Get available store items"""
    with sqlite3.connect(db_manager.db_name) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM store_items ORDER BY category, cost_points')
        items = [dict(row) for row in cursor.fetchall()]

    return {'items': items}

@app.post("/api/store/purchase")
async def purchase_store_item(request: dict, authorization: Optional[str] = Header(None)):
    """Purchase an item from the wellness store"""
    user_id = get_user_from_token(authorization)
    item_id = request.get('item_id')

    # Get item details
    with sqlite3.connect(db_manager.db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT cost_points, name FROM store_items WHERE id = ?', (item_id,))
        item = cursor.fetchone()

        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        cost, name = item

        # Check if user has enough points
        if db_manager.deduct_points(user_id, cost):
            # Record purchase
            cursor.execute('INSERT INTO user_purchases (user_id, item_id) VALUES (?, ?)',
                          (user_id, item_id))
            conn.commit()

            return {
                'success': True,
                'message': f'Successfully purchased {name}!',
                'remaining_points': db_manager.get_user_points(user_id)
            }
        else:
            return {
                'success': False,
                'message': 'Not enough wellness points'
            }

@app.post("/api/buddies/request")
async def send_buddy_request(request: dict, authorization: Optional[str] = Header(None)):
    """Send wellness buddy connection request"""
    user_id = get_user_from_token(authorization)
    buddy_email = request.get('buddy_email')

    result = db_manager.send_buddy_request(user_id, buddy_email)

    if result['success']:
        return JSONResponse(status_code=200, content=result)
    else:
        raise HTTPException(status_code=400, detail=result['message'])

@app.get("/api/buddies")
async def get_buddy_connections(authorization: Optional[str] = Header(None)):
    """Get user's wellness buddy connections"""
    user_id = get_user_from_token(authorization)

    with sqlite3.connect(db_manager.db_name) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get accepted buddies
        cursor.execute('''SELECT uc.id, uc.buddy_id, u.name, uc.accepted_at
            FROM user_connections uc
            JOIN users u ON uc.buddy_id = u.id
            WHERE uc.user_id = ? AND uc.status = 'accepted' ''', (user_id,))
        buddies = [dict(row) for row in cursor.fetchall()]

        # Get pending requests
        cursor.execute('''SELECT uc.id, uc.user_id as requester_id, u.name, uc.created_at
            FROM user_connections uc
            JOIN users u ON uc.user_id = u.id
            WHERE uc.buddy_id = ? AND uc.status = 'pending' ''', (user_id,))
        pending = [dict(row) for row in cursor.fetchall()]

    return {
        'buddies': buddies,
        'pending_requests': pending
    }

@app.post("/api/buddies/{request_id}/accept")
async def accept_buddy(request_id: int, authorization: Optional[str] = Header(None)):
    """Accept a buddy request"""
    user_id = get_user_from_token(authorization)
    result = db_manager.accept_buddy_request(user_id, request_id)

    if result['success']:
        return result
    else:
        raise HTTPException(status_code=400, detail=result['message'])

@app.get("/api/buddies/{buddy_id}/stats")
async def get_buddy_stats(buddy_id: int, authorization: Optional[str] = Header(None)):
    """Get anonymized stats for a wellness buddy"""
    user_id = get_user_from_token(authorization)
    stats = db_manager.get_buddy_stats(user_id, buddy_id)

    if stats['success']:
        return stats
    else:
        raise HTTPException(status_code=403, detail=stats['message'])

@app.get("/api/data/export")
async def export_user_data(authorization: Optional[str] = Header(None)):
    """Export all user data as JSON"""
    user_id = get_user_from_token(authorization)

    data = db_manager.export_user_data(user_id)

    # Return as downloadable JSON
    json_str = json.dumps(data, indent=2, default=str)

    return StreamingResponse(
        iter([json_str]),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=wellness_data_{user_id}.json"
        }
    )

@app.get("/api/data/export/csv")
async def export_health_logs_csv(authorization: Optional[str] = Header(None)):
    """Export health logs as CSV"""
    user_id = get_user_from_token(authorization)

    logs = db_manager.get_health_logs(user_id, days=365)

    # Create CSV
    output = StringIO()
    if logs:
        writer = csv.DictWriter(output, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)

    csv_content = output.getvalue()

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=health_logs_{user_id}.csv"
        }
    )

@app.post("/api/sync/external")
async def sync_external_data(request: dict, authorization: Optional[str] = Header(None)):
    """Mock endpoint for external health tracker sync"""
    user_id = get_user_from_token(authorization)

    # In real implementation, this would validate and sync with Google Fit, Apple Health, etc.
    sync_data = request.get('data', [])

    synced_count = 0
    for entry in sync_data:
        try:
            db_manager.save_health_log(user_id, entry)
            synced_count += 1
        except:
            pass

    return {
        'success': True,
        'synced_entries': synced_count,
        'message': f'Successfully synced {synced_count} entries from external tracker'
    }

@app.post("/api/journal/cbt-analysis")
async def analyze_journal_for_cbt(request: dict, authorization: Optional[str] = Header(None)):
    """Analyze journal entry for CBT patterns"""
    user_id = get_user_from_token(authorization)
    content = request.get('content', '')
    entry_id = request.get('entry_id')

    # Detect negative thought patterns
    patterns = emotion_analyzer.detect_cbt_patterns(content)

    if patterns:
        # Save CBT analysis
        with sqlite3.connect(db_manager.db_name) as conn:
            cursor = conn.cursor()
            for pattern in patterns:
                cursor.execute('''INSERT INTO cbt_thought_records
                    (user_id, journal_entry_id, negative_thought, thought_pattern, reframe_prompt)
                    VALUES (?, ?, ?, ?, ?)''',
                    (user_id, entry_id, content[:200], pattern['pattern'], pattern['reframe_prompt']))
            conn.commit()

    return {
        'patterns_detected': patterns,
        'cbt_guidance': "I noticed some thought patterns that might benefit from reframing. Try the suggested prompts below." if patterns else "Your thoughts seem balanced! Keep up the positive mindset."
    }

@app.post("/api/habits/micro-stack")
async def create_micro_habit(request: dict, authorization: Optional[str] = Header(None)):
    """Create a micro habit stack"""
    user_id = get_user_from_token(authorization)
    goal_type = request.get('goal_type')
    anchor_habit = request.get('anchor_habit')

    suggestion = micro_habit_stacker.suggest_micro_habit(goal_type, anchor_habit)

    if 'anchor' in suggestion:
        # Save to database
        with sqlite3.connect(db_manager.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO micro_habits (user_id, anchor_habit, new_habit)
                VALUES (?, ?, ?)''',
                (user_id, suggestion['anchor'], suggestion['micro_habit']))
            conn.commit()

    return suggestion

@app.get("/api/habits/micro-stack")
async def get_micro_habits(authorization: Optional[str] = Header(None)):
    """Get user's micro habit stacks"""
    user_id = get_user_from_token(authorization)

    with sqlite3.connect(db_manager.db_name) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''SELECT * FROM micro_habits WHERE user_id = ?
            ORDER BY created_at DESC''', (user_id,))
        habits = [dict(row) for row in cursor.fetchall()]

    return {'micro_habits': habits}

@app.post("/api/habits/micro-stack/{habit_id}/complete")
async def complete_micro_habit(habit_id: int, authorization: Optional[str] = Header(None)):
    """Mark micro habit as completed for today"""
    user_id = get_user_from_token(authorization)

    with sqlite3.connect(db_manager.db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('''UPDATE micro_habits
            SET completed_days = completed_days + 1
            WHERE id = ? AND user_id = ?''', (habit_id, user_id))
        conn.commit()

        # Award points
        db_manager.award_points(user_id, 5, "Micro habit completed")

    return {'success': True, 'message': 'Micro habit completed! +5 points'}

@app.post("/api/wellness-vibe/generate")
async def generate_wellness_soundscape(request: dict, authorization: Optional[str] = Header(None)):
    """Generate personalized wellness soundscape based on mood"""
    user_id = get_user_from_token(authorization)
    emotion = request.get('emotion', 'neutral')

    # Mock soundscape generation
    soundscape_configs = {
        'stress': {
            'type': 'Binaural Beats + Ocean Waves',
            'frequency': '432 Hz (calming)',
            'duration': '5 minutes',
            'elements': ['Alpha wave binaural beats', 'Gentle ocean waves', 'Soft wind chimes'],
            'reasoning': 'Binaural beats at 432Hz promote relaxation and alpha brain wave states. Ocean waves provide rhythmic, predictable sounds that calm the nervous system.'
        },
        'anxiety': {
            'type': 'Grounding Soundscape',
            'frequency': '396 Hz (root chakra)',
            'duration': '5 minutes',
            'elements': ['Deep bass tones', 'Forest ambience', 'Gentle rain'],
            'reasoning': '396Hz helps release fear and anxiety. Grounding nature sounds reconnect you to the present moment.'
        },
        'sadness': {
            'type': 'Mood-Lifting Harmony',
            'frequency': '528 Hz (transformation)',
            'duration': '5 minutes',
            'elements': ['Uplifting piano melodies', 'Birdsong', 'Soft strings'],
            'reasoning': '528Hz is associated with DNA repair and positive transformation. Uplifting melodies gradually shift mood.'
        },
        'fatigue': {
            'type': 'Energizing Soundscape',
            'frequency': '417 Hz (energy)',
            'duration': '5 minutes',
            'elements': ['Rhythmic percussion', 'Upbeat nature sounds', 'Wind instruments'],
            'reasoning': '417Hz facilitates change and removes negative energy. Rhythmic elements boost alertness.'
        }
    }

    soundscape = soundscape_configs.get(emotion, soundscape_configs['stress'])

    # Save to database
    with sqlite3.connect(db_manager.db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO wellness_soundscapes (user_id, emotion, soundscape_config)
            VALUES (?, ?, ?)''',
            (user_id, emotion, json.dumps(soundscape)))
        conn.commit()

    return {
        'soundscape': soundscape,
        'listen_url': f'/api/wellness-vibe/play/{emotion}',  # Mock URL
        'message': f'Your personalized "{soundscape["type"]}" is ready!'
    }

@app.get("/api/challenges/group")
async def get_group_challenges():
    """Get available group challenges"""
    with sqlite3.connect(db_manager.db_name) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''SELECT cg.*, u.name as creator_name,
            (SELECT COUNT(*) FROM group_members WHERE group_id = cg.id) as member_count
            FROM challenge_groups cg
            JOIN users u ON cg.created_by = u.id
            WHERE cg.end_date >= date('now')
            ORDER BY created_at DESC''')
        challenges = [dict(row) for row in cursor.fetchall()]

    return {'group_challenges': challenges}

@app.post("/api/challenges/group/create")
async def create_group_challenge(request: dict, authorization: Optional[str] = Header(None)):
    """Create a new group challenge"""
    user_id = get_user_from_token(authorization)

    name = request.get('name')
    description = request.get('description')
    goal_type = request.get('goal_type')
    target_value = request.get('target_value')
    duration_days = request.get('duration_days', 30)

    end_date = (datetime.date.today() + datetime.timedelta(days=duration_days)).isoformat()

    with sqlite3.connect(db_manager.db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO challenge_groups
            (name, description, goal_type, target_value, created_by, end_date)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (name, description, goal_type, target_value, user_id, end_date))

        group_id = cursor.lastrowid

        # Auto-join creator
        cursor.execute('INSERT INTO group_members (group_id, user_id) VALUES (?, ?)',
                      (group_id, user_id))
        conn.commit()

    return {
        'success': True,
        'group_id': group_id,
        'message': f'Group challenge "{name}" created!'
    }

@app.post("/api/challenges/group/{group_id}/join")
async def join_group_challenge(group_id: int, authorization: Optional[str] = Header(None)):
    """Join a group challenge"""
    user_id = get_user_from_token(authorization)

    with sqlite3.connect(db_manager.db_name) as conn:
        cursor = conn.cursor()

        # Check if already member
        cursor.execute('SELECT id FROM group_members WHERE group_id = ? AND user_id = ?',
                      (group_id, user_id))

        if cursor.fetchone():
            return {'success': False, 'message': 'Already a member of this challenge'}

        cursor.execute('INSERT INTO group_members (group_id, user_id) VALUES (?, ?)',
                      (group_id, user_id))
        conn.commit()

    return {'success': True, 'message': 'Successfully joined the challenge!'}

@app.get("/api/challenges/group/{group_id}/leaderboard")
async def get_group_leaderboard(group_id: int):
    """Get leaderboard for a group challenge"""
    with sqlite3.connect(db_manager.db_name) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get challenge details
        cursor.execute('SELECT goal_type FROM challenge_groups WHERE id = ?', (group_id,))
        challenge = cursor.fetchone()

        if not challenge:
            raise HTTPException(status_code=404, detail="Challenge not found")

        goal_type = challenge['goal_type']

        # Get member stats (this is simplified - would need more complex aggregation in production)
        cursor.execute('''SELECT gm.user_id, u.name,
            COUNT(DISTINCT hl.log_date) as days_logged,
            AVG(CASE
                WHEN ? = 'steps' THEN hl.steps
                WHEN ? = 'water' THEN hl.water_intake
                WHEN ? = 'sleep' THEN hl.sleep_hours
                ELSE 0
            END) as avg_value
            FROM group_members gm
            JOIN users u ON gm.user_id = u.id
            LEFT JOIN health_logs hl ON hl.user_id = gm.user_id
            WHERE gm.group_id = ?
            GROUP BY gm.user_id, u.name
            ORDER BY avg_value DESC, days_logged DESC''',
            (goal_type, goal_type, goal_type, group_id))

        leaderboard = [dict(row) for row in cursor.fetchall()]

    return {'leaderboard': leaderboard}
# ============================================================================
# HTML FRONTEND
# ============================================================================


HTML_FRONTEND = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Global Wellness Platform v7.0</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --primary: #2d5a87;
            --accent: #00c9a7;
            --success: #4285f4;
            --warning: #ffa726;
            --error: #ef5350;
            --bg-dark: #0f1419;
            --bg-light: #1a202c;
            --glass-bg: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.15);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-light) 50%, var(--bg-dark) 100%);
            min-height: 100vh;
            color: white;
            transition: all 0.3s;
        }

        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }

        .header {
            background: linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%);
            padding: 40px; border-radius: 25px; margin-bottom: 40px;
            text-align: center; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }

        .daily-quote {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border); padding: 25px;
            border-radius: 15px; margin-bottom: 30px;
            text-align: center; font-style: italic; font-size: 1.1rem; line-height: 1.6;
        }

        .screen { display: none; animation: fadeIn 0.5s ease-in; }
        .screen.active { display: block; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .menu-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px; margin: 40px 0;
        }

        .menu-btn {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 2px solid var(--glass-border); padding: 30px;
            border-radius: 20px; color: white; font-size: 1.1rem;
            cursor: pointer; transition: all 0.4s; display: flex;
            flex-direction: column; align-items: center; gap: 15px;
            text-align: center;
        }

        .menu-btn:hover {
            transform: translateY(-8px); border-color: var(--accent);
            box-shadow: 0 20px 40px rgba(0, 201, 167, 0.25);
        }

        .form-container {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border); padding: 40px;
            border-radius: 25px; max-width: 550px; margin: 0 auto;
        }

        .form-group { margin-bottom: 25px; }
        .form-group label { display: block; margin-bottom: 10px; font-weight: 600; }

        .form-group input, .form-group select, .form-group textarea {
            width: 100%; padding: 14px; border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px; background: rgba(255, 255, 255, 0.06);
            color: white; font-size: 1rem; font-family: inherit;
        }

        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none; border-color: var(--accent);
            background: rgba(0, 201, 167, 0.1);
            box-shadow: 0 0 20px rgba(0, 201, 167, 0.2);
        }

        .btn {
            background: linear-gradient(135deg, #00c9a7 0%, #00a085 100%);
            color: white; border: none; padding: 16px 35px;
            border-radius: 12px; font-size: 1rem; font-weight: 700;
            cursor: pointer; transition: all 0.3s; box-shadow: 0 10px 30px rgba(0, 201, 167, 0.2);
        }

        .btn:hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(0, 201, 167, 0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .back-btn {
            background: var(--glass-bg); border: 1px solid var(--glass-border);
            padding: 12px 25px; border-radius: 12px; color: white;
            cursor: pointer; margin-bottom: 30px; font-weight: 600;
            transition: all 0.3s; display: inline-flex; align-items: center; gap: 8px;
        }

        .back-btn:hover { background: rgba(0, 201, 167, 0.1); border-color: var(--accent); }

        .stat-card {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border); padding: 30px;
            border-radius: 20px; text-align: center; transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .stat-card:hover { transform: translateY(-5px); border-color: var(--accent); }

        .stat-number {
            font-size: 3rem; font-weight: 800;
            background: linear-gradient(135deg, #00c9a7 0%, #00a085 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 15px 0;
        }

        .chat-container {
            background: var(--glass-bg); border: 1px solid var(--glass-border);
            border-radius: 25px; height: 700px; display: flex;
            flex-direction: column; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .chat-header {
            background: linear-gradient(135deg, #00c9a7 0%, #00a085 100%);
            padding: 25px; border-radius: 25px 25px 0 0; font-weight: 700;
        }

        .chat-messages { flex: 1; overflow-y: auto; padding: 25px; }

        .message {
            display: flex; gap: 15px; margin-bottom: 20px; animation: messageIn 0.3s ease;
        }

        @keyframes messageIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user { flex-direction: row-reverse; }

        .message-avatar {
            width: 45px; height: 45px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; font-weight: 700;
        }

        .message.bot .message-avatar { background: linear-gradient(135deg, #00c9a7 0%, #00a085 100%); }
        .message.user .message-avatar { background: linear-gradient(135deg, #4285f4 0%, #1a73e8 100%); }

        .message-content {
            background: rgba(255, 255, 255, 0.1); padding: 15px 20px;
            border-radius: 18px; max-width: 65%; line-height: 1.6; word-wrap: break-word;
        }

        .message.user .message-content { background: rgba(66, 133, 244, 0.2); }

        .chat-input-container {
            padding: 20px; border-top: 1px solid var(--glass-border);
            display: flex; gap: 12px;
        }

        .chat-input {
            flex: 1; background: rgba(255, 255, 255, 0.08);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px; padding: 14px; color: white; font-size: 0.95rem;
            transition: all 0.3s;
        }

        .chat-input:focus { outline: none; border-color: var(--accent);
            background: rgba(0, 201, 167, 0.08);
        }

        .dashboard-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px; margin-bottom: 40px;
        }

        .chart-container {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border); padding: 30px;
            border-radius: 20px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .health-tracker {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border); padding: 35px;
            border-radius: 20px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .tracker-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px; margin-top: 25px;
        }

        .tracker-input {
            background: rgba(255, 255, 255, 0.06);
            padding: 18px; border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s;
        }

        .tracker-input:hover { border-color: var(--accent); }

        .tracker-input label {
            display: block; font-size: 0.9rem; margin-bottom: 10px;
            opacity: 0.85; font-weight: 600;
        }

        .tracker-input input, .tracker-input select {
            width: 100%; padding: 10px; border: none;
            border-radius: 8px; background: rgba(255, 255, 255, 0.1);
            color: white; font-size: 0.95rem;
        }

        .goal-card {
            background: var(--glass-bg); border: 1px solid var(--glass-border);
            padding: 20px; border-radius: 15px; position: relative;
            margin-bottom: 15px;
        }

        .goal-progress { width: 100%; height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px; margin-top: 15px; overflow: hidden;
        }

        .goal-progress-bar {
            height: 100%; background: linear-gradient(90deg, var(--accent), var(--success));
            border-radius: 10px; transition: width 0.5s ease;
        }

        .streak-badge {
            position: absolute; top: 10px; right: 10px;
            background: linear-gradient(135deg, #ffa726, #ff6f00);
            padding: 5px 12px; border-radius: 20px;
            font-size: 0.8rem; font-weight: bold;
        }

        .meditation-card {
            background: linear-gradient(135deg, rgba(0, 201, 167, 0.15) 0%, rgba(0, 160, 133, 0.15) 100%);
            padding: 25px; border-radius: 15px; margin-bottom: 20px;
            border: 1px solid rgba(0, 201, 167, 0.3);
            cursor: pointer; transition: all 0.3s;
        }

        .meditation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 201, 167, 0.25);
        }

        .meditation-card h3 { margin-bottom: 10px; color: var(--accent); }

        .journal-entry {
            background: rgba(255, 255, 255, 0.06); padding: 20px;
            border-radius: 15px; margin-bottom: 15px;
            border-left: 4px solid var(--accent);
            transition: all 0.3s;
        }

        .journal-entry:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(5px);
        }

        .journal-entry h4 { margin-bottom: 8px; color: var(--accent); }
        .journal-entry small { opacity: 0.6; }
        .journal-entry p { margin-top: 10px; opacity: 0.85; }

        .challenge-card {
            background: rgba(255, 255, 255, 0.08); padding: 20px;
            border-radius: 15px; margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex; justify-content: space-between;
            align-items: center; transition: all 0.3s;
        }

        .challenge-card:hover {
            border-color: var(--accent);
            transform: translateX(5px);
        }

        .badge-container {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 15px; margin-top: 20px;
        }

        .badge {
            background: rgba(255, 255, 255, 0.08);
            border: 2px solid var(--accent); padding: 15px;
            border-radius: 12px; text-align: center; font-size: 2rem;
            cursor: pointer; transition: all 0.3s;
        }

        .badge:hover {
            transform: scale(1.1) rotateZ(5deg);
            box-shadow: 0 10px 30px rgba(0, 201, 167, 0.3);
        }

        .tabs {
            display: flex; gap: 15px; margin-bottom: 30px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            flex-wrap: wrap;
        }

        .tab-btn {
            background: none; border: none; color: rgba(255, 255, 255, 0.6);
            padding: 12px 20px; cursor: pointer; font-weight: 600;
            border-bottom: 3px solid transparent; transition: all 0.3s;
        }

        .tab-btn.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        .tab-btn:hover { color: white; }

        .insight-box {
            background: linear-gradient(135deg, rgba(0, 201, 167, 0.15) 0%, rgba(66, 133, 244, 0.15) 100%);
            padding: 25px; border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 25px; line-height: 1.8;
        }

        .insight-box h3 { margin-bottom: 15px; color: var(--accent); }

        .plan-selector-btn, .tracker-selector-btn {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 2px solid var(--glass-border);
            padding: 20px 40px;
            border-radius: 15px;
            color: white;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            min-width: 200px;
        }

        .plan-selector-btn i, .tracker-selector-btn i {
            font-size: 2rem;
        }

        .plan-selector-btn:hover, .tracker-selector-btn:hover {
            transform: translateY(-5px);
            border-color: var(--accent);
            box-shadow: 0 15px 40px rgba(0, 201, 167, 0.3);
        }

        .plan-selector-btn.active, .tracker-selector-btn.active {
            background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
            border-color: var(--accent);
            box-shadow: 0 10px 30px rgba(0, 201, 167, 0.4);
        }

        .plan-section, .tracker-section {
            display: none;
            animation: fadeIn 0.5s ease-in;
        }

        .plan-section.active, .tracker-section.active {
            display: block;
        }

        .prompt-btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 12px 18px;
            border-radius: 10px;
            color: white;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s;
            text-align: left;
        }

        .prompt-btn:hover {
            background: rgba(0, 201, 167, 0.2);
            border-color: var(--accent);
            transform: translateY(-2px);
        }

        .health-log-card {
            background: rgba(255, 255, 255, 0.06);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 4px solid var(--accent);
            transition: all 0.3s;
        }

        .health-log-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(5px);
        }

        .health-log-card h4 {
            margin-bottom: 15px;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .health-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }

        .metric-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s;
        }

        .metric-item:hover {
            background: rgba(0, 201, 167, 0.1);
            transform: scale(1.05);
        }

        .metric-item .metric-label {
            font-size: 0.85rem;
            opacity: 0.7;
            margin-bottom: 8px;
        }

        .metric-item .metric-value {
            font-size: 1.4rem;
            font-weight: bold;
            color: var(--accent);
        }

        .log-notes {
            margin-top: 15px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }

        canvas { max-height: 350px; }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); }
        ::-webkit-scrollbar-thumb {
            background: var(--accent); border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #00a085; }

        @media (max-width: 768px) {
            .container { padding: 15px; }
            .menu-grid { grid-template-columns: 1fr; }
            .dashboard-grid { grid-template-columns: 1fr; }
            .tracker-grid { grid-template-columns: 1fr; }
            .message-content { max-width: 85%; }
            .chat-container { height: 500px; }
            .plan-selector-btn, .tracker-selector-btn {
                min-width: 150px;
                padding: 15px 30px;
                font-size: 1rem;
            }
            .health-metrics {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        /* Dashboard Styles */
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 25px;
            margin-bottom: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .dashboard-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                45deg,
                transparent 30%,
                rgba(255, 255, 255, 0.1) 50%,
                transparent 70%
            );
            animation: shimmer 3s infinite;
        }

        .dashboard-header h2 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .dashboard-header p {
            font-size: 1.2rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }

        .stat-card-enhanced {
            background: linear-gradient(135deg, var(--glass-bg) 0%, rgba(255, 255, 255, 0.12) 100%);
            backdrop-filter: blur(20px);
            border: 2px solid var(--glass-border);
            padding: 35px;
            border-radius: 25px;
            text-align: center;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .stat-card-enhanced::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }

        .stat-card-enhanced:hover::before {
            left: 100%;
        }

        .stat-card-enhanced:hover {
            transform: translateY(-10px) scale(1.02);
            border-color: var(--accent);
            box-shadow: 0 20px 60px rgba(0, 201, 167, 0.3);
        }

        .stat-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 20px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 10px 30px rgba(0, 201, 167, 0.3);
        }

        .stat-number-enhanced {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00c9a7 0%, #4285f4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 15px 0;
            line-height: 1;
        }

        .stat-label {
            font-size: 1rem;
            opacity: 0.8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-trend {
            margin-top: 10px;
            font-size: 0.9rem;
            padding: 5px 15px;
            border-radius: 20px;
            display: inline-block;
        }

        .trend-up {
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
        }

        .progress-section {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 2px solid var(--glass-border);
            padding: 35px;
            border-radius: 25px;
            margin-bottom: 40px;
        }

        .progress-item {
            margin-bottom: 25px;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .progress-bar-container {
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            border-radius: 10px;
            transition: width 1s ease-out;
            position: relative;
            overflow: hidden;
        }

        .progress-bar-fill::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }

        .charts-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .chart-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 2px solid var(--glass-border);
            padding: 35px;
            border-radius: 25px;
            transition: all 0.4s;
            position: relative;
        }

        .chart-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            border-color: var(--accent);
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }

        .chart-title {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .chart-badge {
            background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }

        .insight-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
            border: 2px solid rgba(102, 126, 234, 0.3);
            padding: 30px;
            border-radius: 20px;
            transition: all 0.3s;
        }

        .insight-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        }

        .insight-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }

        .insight-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: var(--accent);
        }

        .insight-text {
            line-height: 1.6;
            opacity: 0.9;
        }

        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }

        .quick-action-btn {
            background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
            border: none;
            padding: 20px;
            border-radius: 15px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-size: 1rem;
        }

        .quick-action-btn:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 201, 167, 0.4);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }

            .charts-section {
                grid-template-columns: 1fr;
            }

            .insights-grid {
                grid-template-columns: 1fr;
            }

            .dashboard-header h2 {
                font-size: 1.8rem;
            }

            .stat-number-enhanced {
                font-size: 2.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-heartbeat"></i> Enhanced Global Wellness Platform</h1>
            <p>🌍 Your Complete AI Health & Wellness Companion v7.0</p>

            <!-- NEW: Points Display -->
            <div id="points-display" style="display: none; margin-top: 15px;">
                <div style="display: inline-block; background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 20px;">
                    <span style="font-size: 1.2rem;">💎 <span id="points-value">0</span> Wellness Points</span>
                    <span style="margin-left: 15px; opacity: 0.8;" id="user-rank">Explorer</span>
                </div>
            </div>
        </div>

        <!-- NEW: Nudges Banner -->
        <div id="nudges-banner" style="display: none;"></div>

        <div id="main-menu" class="screen active">
            <div class="daily-quote">
                <i class="fas fa-quote-left"></i>
                <p id="daily-quote-text">Loading inspirational quote...</p>
                <i class="fas fa-quote-right"></i>
            </div>
            <div class="menu-grid" id="menu-options"></div>
        </div>

        <div id="register-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <div class="form-container">
                <h2 style="text-align: center; margin-bottom: 30px;">Create Account</h2>
                <form id="register-form">
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="reg-email" required>
                    </div>
                    <div class="form-group">
                        <label>Password</label>
                        <input type="password" id="reg-password" required>
                    </div>
                    <div class="form-group">
                        <label>Confirm Password</label>
                        <input type="password" id="reg-confirm-password" required>
                    </div>
                    <div class="form-group">
                        <label>Full Name</label>
                        <input type="text" id="reg-name" required>
                    </div>
                    <div class="form-group">
                        <label>Age (Optional)</label>
                        <input type="number" id="reg-age" min="13" max="120">
                    </div>
                    <button type="submit" class="btn">Register</button>
                </form>
            </div>
        </div>

        <div id="login-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <div class="form-container">
                <h2 style="text-align: center; margin-bottom: 30px;">Login</h2>
                <form id="login-form">
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="login-email" required>
                    </div>
                    <div class="form-group">
                        <label>Password</label>
                        <input type="password" id="login-password" required>
                    </div>
                    <button type="submit" class="btn">Login</button>
                </form>
            </div>
        </div>

        <div id="chat-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <div class="chat-container">
                <div class="chat-header">
                    <h3><i class="fas fa-robot"></i> AI Health Assistant</h3>
                    <small>Multilingual • Emotion-aware • Crisis Support</small>
                </div>
                <div class="chat-messages" id="chat-messages">
                    <div class="message bot">
                        <div class="message-avatar"><i class="fas fa-robot"></i></div>
                        <div class="message-content">👋 Hello! I'm your AI wellness companion. Chat in any language and ask about symptoms, health tips, fitness, and more!</div>
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chat-input" placeholder="Type your health question..." onkeypress="handleChatKeyPress(event)">
                    <button class="btn" style="width: auto; padding: 14px 28px;" onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
                </div>
            </div>
        </div>

        <div id="health-journal-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">📊 Health Tracking & Wellness Journal</h2>

            <div style="display: flex; gap: 20px; justify-content: center; margin-bottom: 40px; flex-wrap: wrap;">
                <button class="tracker-selector-btn active" id="tracker-selector" onclick="switchTrackerType('tracker')">
                    <i class="fas fa-heartbeat"></i>
                    <span>Health Tracker</span>
                </button>
                <button class="tracker-selector-btn" id="journal-selector-btn" onclick="switchTrackerType('journal')">
                    <i class="fas fa-pen-fancy"></i>
                    <span>Journal</span>
                </button>
            </div>

            <div id="tracker-section" class="tracker-section active">
                <div class="health-tracker">
                    <h3 style="margin-bottom: 20px;"><i class="fas fa-chart-line"></i> Log Your Daily Health Metrics</h3>
                    <form id="health-log-form">
                        <div class="tracker-grid">
                            <div class="tracker-input">
                                <label><i class="fas fa-walking"></i> Steps</label>
                                <input type="number" id="log-steps" min="0" placeholder="e.g., 10000">
                            </div>
                            <div class="tracker-input">
                                <label><i class="fas fa-fire"></i> Calories</label>
                                <input type="number" id="log-calories" min="0" placeholder="e.g., 2000">
                            </div>
                            <div class="tracker-input">
                                <label><i class="fas fa-bed"></i> Sleep (hours)</label>
                                <input type="number" id="log-sleep" min="0" max="24" step="0.5" placeholder="e.g., 7.5">
                            </div>
                            <div class="tracker-input">
                                <label><i class="fas fa-tint"></i> Water (liters)</label>
                                <input type="number" id="log-water" min="0" step="0.1" placeholder="e.g., 2.5">
                            </div>
                            <div class="tracker-input">
                                <label><i class="fas fa-smile"></i> Mood</label>
                                <select id="log-mood">
                                    <option value="">Select mood</option>
                                    <option value="excellent">😄 Excellent</option>
                                    <option value="good">😊 Good</option>
                                    <option value="okay">😐 Okay</option>
                                    <option value="bad">😔 Bad</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-group" style="margin-top: 20px;">
                            <label><i class="fas fa-sticky-note"></i> Notes (Optional)</label>
                            <textarea id="log-notes" placeholder="Any observations about your health today..."
                                      style="width: 100%; min-height: 100px; padding: 14px; border: 2px solid rgba(255, 255, 255, 0.1);
                                      border-radius: 12px; background: rgba(255, 255, 255, 0.06); color: white; font-size: 1rem;
                                      font-family: inherit; resize: vertical;"></textarea>
                        </div>
                        <button type="submit" class="btn" style="margin-top: 20px; width: 100%;">
                            <i class="fas fa-save"></i> Save Health Log
                        </button>
                    </form>
                </div>

                <div style="margin-top: 40px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h3><i class="fas fa-history"></i> Recent Health Logs</h3>
                        <button class="btn" style="padding: 10px 20px; width: auto;" onclick="loadHealthLogsDisplay()">
                            <i class="fas fa-sync"></i> Refresh
                        </button>
                    </div>
                    <div id="health-logs-display">
                        <p style="text-align: center; opacity: 0.6;">Your health logs will appear here...</p>
                    </div>
                </div>
            </div>

            <div id="journal-section-content" class="tracker-section">
                <div class="health-tracker">
                    <h3 style="margin-bottom: 20px;"><i class="fas fa-book-open"></i> Write Your Wellness Journal</h3>
                    <form id="journal-form">
                        <div class="form-group">
                            <label><i class="fas fa-heading"></i> Title</label>
                            <input type="text" id="journal-title" placeholder="Give your entry a title..." required>
                        </div>
                        <div class="form-group" style="margin-top: 20px;">
                            <label><i class="fas fa-pen"></i> Your Thoughts</label>
                            <textarea id="journal-content" placeholder="Express your feelings, thoughts, and experiences..."
                                      required style="width: 100%; min-height: 250px; padding: 14px;
                                      border: 2px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
                                      background: rgba(255, 255, 255, 0.06); color: white; font-size: 1rem;
                                      font-family: inherit; resize: vertical; line-height: 1.8;"></textarea>
                        </div>

                        <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; margin-top: 20px;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                                <i class="fas fa-lightbulb" style="color: var(--accent); font-size: 1.2rem;"></i>
                                <label style="margin: 0; font-weight: 600;">Need inspiration? Try these prompts:</label>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                <button type="button" class="prompt-btn" onclick="usePrompt('gratitude')">
                                    🙏 What I'm grateful for
                                </button>
                                <button type="button" class="prompt-btn" onclick="usePrompt('achievement')">
                                    ⭐ Today's achievements
                                </button>
                                <button type="button" class="prompt-btn" onclick="usePrompt('challenge')">
                                    💪 Challenges I faced
                                </button>
                                <button type="button" class="prompt-btn" onclick="usePrompt('lesson')">
                                    📚 Lessons learned
                                </button>
                                <button type="button" class="prompt-btn" onclick="usePrompt('goals')">
                                    🎯 Tomorrow's goals
                                </button>
                                <button type="button" class="prompt-btn" onclick="usePrompt('reflection')">
                                    🤔 Daily reflection
                                </button>
                            </div>
                        </div>

                        <button type="submit" class="btn" style="margin-top: 20px; width: 100%;">
                            <i class="fas fa-save"></i> Save Journal Entry
                        </button>
                    </form>
                </div>

                <div style="margin-top: 40px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h3><i class="fas fa-history"></i> Recent Journal Entries</h3>
                        <button class="btn" style="padding: 10px 20px; width: auto;" onclick="loadJournalDisplay()">
                            <i class="fas fa-sync"></i> Refresh
                        </button>
                    </div>
                    <div id="journal-entries-display">
                        <p style="text-align: center; opacity: 0.6;">Your journal entries will appear here...</p>
                    </div>
                </div>
            </div>
        </div>

        <div id="dashboard-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')">
                <i class="fas fa-arrow-left"></i> Back
            </button>

            <div class="dashboard-header">
                <h2>🌟 Your Wellness Journey</h2>
                <p>Tracking your path to better health</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card-enhanced">
                    <div class="stat-icon">💬</div>
                    <div class="stat-number-enhanced" id="enhanced-chats">...</div>
                    <div class="stat-label">Conversations</div>
                    <div class="stat-trend trend-up">📊 AI Interactions</div>
                </div>

                <div class="stat-card-enhanced">
                    <div class="stat-icon">❤️</div>
                    <div class="stat-number-enhanced" id="enhanced-health-logs">...</div>
                    <div class="stat-label">Health Logs</div>
                    <div class="stat-trend trend-up">📈 Tracked Days</div>
                </div>

                <div class="stat-card-enhanced">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-number-enhanced" id="enhanced-goals">...</div>
                    <div class="stat-label">Active Goals</div>
                    <div class="stat-trend trend-up">🔥 Keep going!</div>
                </div>

                <div class="stat-card-enhanced">
                    <div class="stat-icon">🔥</div>
                    <div class="stat-number-enhanced" id="enhanced-streak">...</div>
                    <div class="stat-label">Day Streak</div>
                    <div class="stat-trend trend-up">⭐ Amazing!</div>
                </div>
            </div>

            <div class="progress-section">
                <h3 style="margin-bottom: 25px; font-size: 1.5rem;">📊 Weekly Progress</h3>

                <div class="progress-item">
                    <div class="progress-label">
                        <span>Daily Water Goal</span>
                        <span id="water-progress-percent">0%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" id="water-progress-bar" style="width: 0%"></div>
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <span>Exercise Minutes</span>
                        <span id="exercise-progress-percent">0%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" id="exercise-progress-bar" style="width: 0%"></div>
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <span>Sleep Quality</span>
                        <span id="sleep-progress-percent">0%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" id="sleep-progress-bar" style="width: 0%"></div>
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <span>Step Goals</span>
                        <span id="steps-progress-percent">0%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" id="steps-progress-bar" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <div class="charts-section">
                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">
                            😊 Mood Analysis
                        </div>
                        <div class="chart-badge">Last 7 days</div>
                    </div>
                    <canvas id="enhanced-sentiment-chart" style="height: 300px;"></canvas>
                </div>

                <div class="chart-card">
                    <div class="chart-header">
                        <div class="chart-title">
                            🎭 Emotion Trends
                        </div>
                        <div class="chart-badge">This month</div>
                    </div>
                    <canvas id="enhanced-emotion-chart" style="height: 300px;"></canvas>
                </div>
            </div>

            <div class="chart-card" style="margin-bottom: 30px;">
                <div class="chart-header">
                    <div class="chart-title">
                        📈 Activity Timeline
                    </div>
                    <div class="chart-badge">Last 14 days</div>
                </div>
                <canvas id="activity-timeline-chart" style="height: 250px;"></canvas>
            </div>

            <div class="insights-grid">
                <div class="insight-card">
                    <div class="insight-icon">💪</div>
                    <div class="insight-title">Health Score</div>
                    <div class="insight-text" id="health-score-insight">
                        Loading your personalized score...
                    </div>
                </div>

                <div class="insight-card">
                    <div class="insight-icon">⚡</div>
                    <div class="insight-title">Top Strength</div>
                    <div class="insight-text" id="top-strength-insight">
                        Keep tracking to see your strengths!
                    </div>
                </div>

                <div class="insight-card">
                    <div class="insight-icon">🎯</div>
                    <div class="insight-title">Improvement Area</div>
                    <div class="insight-text" id="improvement-area-insight">
                        Areas to focus on will appear here.
                    </div>
                </div>
            </div>

            <div style="background: linear-gradient(135deg, rgba(0, 201, 167, 0.15) 0%, rgba(66, 133, 244, 0.15) 100%); padding: 30px; border-radius: 20px; border: 2px solid rgba(0, 201, 167, 0.2); margin-bottom: 30px;">
                <h3 style="margin-bottom: 20px; color: var(--accent); font-size: 1.3rem;">
                    <i class="fas fa-lightbulb"></i> Your Personalized Recommendations
                </h3>
                <div id="recommendations-list">
                    <p style="opacity: 0.7; text-align: center;">Start logging health data to receive personalized tips!</p>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div style="background: rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 15px; border-left: 4px solid #FF6B6B;">
                    <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 10px;">Avg. Sleep</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #FF6B6B;" id="avg-sleep-stat">-- hrs</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 8px;">Last 30 days</div>
                </div>

                <div style="background: rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 15px; border-left: 4px solid #4ECDC4;">
                    <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 10px;">Avg. Steps</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #4ECDC4;" id="avg-steps-stat">-- steps</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 8px;">Daily average</div>
                </div>

                <div style="background: rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 15px; border-left: 4px solid #45B7D1;">
                    <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 10px;">Avg. Water</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #45B7D1;" id="avg-water-stat">-- L</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 8px;">Daily intake</div>
                </div>

                <div style="background: rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 15px; border-left: 4px solid #FFA500;">
                    <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 10px;">Mood Trend</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #FFA500;" id="mood-trend-stat">😊</div>
                    <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 8px;" id="mood-trend-text">See your patterns</div>
                </div>
            </div>

            <div class="quick-actions">
                <button class="quick-action-btn" onclick="showScreen('health-journal-screen')">
                    <i class="fas fa-plus"></i> Log Health Data
                </button>
                <button class="quick-action-btn" onclick="showScreen('goals-screen')">
                    <i class="fas fa-target"></i> Set New Goal
                </button>
                <button class="quick-action-btn" onclick="showScreen('meditation-screen')">
                    <i class="fas fa-spa"></i> Meditate Now
                </button>
                <button class="quick-action-btn" onclick="showScreen('chat-screen')">
                    <i class="fas fa-robot"></i> Ask AI Assistant
                </button>
            </div>

            <div id="no-data-message" style="display: none; text-align: center; padding: 40px; opacity: 0.6;">
                <h3>📊 Start Your Wellness Journey</h3>
                <p>No health data found. Begin by logging your health metrics or having a chat!</p>
            </div>
        </div>


        <div id="goals-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🎯 Wellness Goals & Streaks</h2>
            <div class="form-container" style="max-width: 700px; margin-bottom: 30px;">
                <h3 style="margin-bottom: 20px;">Create New Goal</h3>
                <form id="goal-form">
                    <div class="form-group">
                        <label>Goal Type</label>
                        <select id="goal-type" required>
                            <option value="">Select goal type</option>
                            <option value="water">Water Intake (liters/day)</option>
                            <option value="steps">Steps (per day)</option>
                            <option value="sleep">Sleep (hours/night)</option>
                            <option value="meditation">Meditation (minutes/day)</option>
                            <option value="exercise">Exercise (minutes/day)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Target Value</label>
                        <input type="number" id="goal-target" step="0.1" required>
                    </div>
                    <div class="form-group">
                        <label>Description (Optional)</label>
                        <input type="text" id="goal-description" placeholder="e.g., Drink 3L water daily">
                    </div>
                    <button type="submit" class="btn">Create Goal</button>
                </form>
            </div>
            <h3 style="margin-bottom: 20px;">Active Goals</h3>
            <div id="goals-list"></div>
        </div>

        <div id="nutrition-fitness-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🏋️ Nutrition & Fitness Hub</h2>

            <div style="display: flex; gap: 20px; justify-content: center; margin-bottom: 40px; flex-wrap: wrap;">
                <button class="plan-selector-btn active" id="diet-selector" onclick="switchPlanType('diet')">
                    <i class="fas fa-utensils"></i>
                    <span>Diet Plan</span>
                </button>
                <button class="plan-selector-btn" id="fitness-selector" onclick="switchPlanType('fitness')">
                    <i class="fas fa-dumbbell"></i>
                    <span>Fitness Plan</span>
                </button>
            </div>

            <div id="diet-plan-section" class="plan-section active">
                <div class="form-container" style="max-width: 700px; margin-bottom: 30px;">
                    <h3 style="margin-bottom: 20px;">🍽️ Generate Your AI Diet Plan</h3>
                    <form id="diet-form">
                        <div class="form-group">
                            <label>Diet Preference</label>
                            <select id="diet-preference" required>
                                <option value="vegetarian">Vegetarian</option>
                                <option value="non-veg">Non-Vegetarian</option>
                                <option value="vegan">Vegan</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Health Goal</label>
                            <select id="health-goal" required>
                                <option value="weight_loss">Weight Loss</option>
                                <option value="muscle_gain">Muscle Gain</option>
                                <option value="maintenance">Maintenance</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Daily Calorie Target</label>
                            <input type="number" id="calories-target" value="2000" required>
                        </div>
                        <button type="submit" class="btn">Generate AI Diet Plan</button>
                    </form>
                </div>
                <div id="diet-plan-display"></div>
            </div>

            <div id="fitness-plan-section" class="plan-section">
                <div class="form-container" style="max-width: 700px; margin-bottom: 30px;">
                    <h3 style="margin-bottom: 20px;">💪 Generate Your AI Workout Plan</h3>
                    <form id="fitness-form">
                        <div class="form-group">
                            <label>Fitness Level</label>
                            <select id="fitness-level" required>
                                <option value="beginner">Beginner</option>
                                <option value="intermediate">Intermediate</option>
                                <option value="advanced">Advanced</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Fitness Goal</label>
                            <select id="fitness-goal" required>
                                <option value="weight_loss">Weight Loss</option>
                                <option value="muscle_gain">Muscle Gain</option>
                                <option value="maintenance">General Fitness</option>
                            </select>
                        </div>
                        <button type="submit" class="btn">Generate AI Fitness Plan</button>
                    </form>
                </div>
                <div id="fitness-plan-display"></div>
            </div>
        </div>

        <div id="community-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">👥 Wellness Community</h2>
            <div class="form-container" style="max-width: 700px; margin-bottom: 30px;">
                <h3 style="margin-bottom: 20px;">Share Your Journey</h3>
                <form id="community-form">
                    <div class="form-group">
                        <label>Title</label>
                        <input type="text" id="post-title" required>
                    </div>
                    <div class="form-group">
                        <label>Content</label>
                        <input type="text" id="post-content" required placeholder="Share your wellness tips...">
                    </div>
                    <div class="form-group">
                        <label><input type="checkbox" id="post-anonymous"> Post Anonymously</label>
                    </div>
                    <button type="submit" class="btn">Post</button>
                </form>
            </div>
            <h3 style="margin-bottom: 20px;">Recent Posts</h3>
            <div id="community-posts"></div>
        </div>

        <div id="meditation-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🧘 Mindfulness & Meditation Hub</h2>
            <div class="meditation-card" onclick="startMeditation('breathing')">
                <h3><i class="fas fa-wind"></i> Breathing Exercise</h3>
                <p>5-minute guided breathing to calm your mind. Perfect for anxiety and stress relief.</p>
            </div>
            <div class="meditation-card" onclick="startMeditation('body_scan')">
                <h3><i class="fas fa-person"></i> Body Scan Meditation</h3>
                <p>Progressive relaxation from head to toe. Great before sleep.</p>
            </div>
            <div class="meditation-card" onclick="startMeditation('gratitude')">
                <h3><i class="fas fa-hands-praying"></i> Gratitude Practice</h3>
                <p>Shift your mindset with gratitude. Daily practice transforms your perspective.</p>
            </div>
            <div class="meditation-card" onclick="startMeditation('stress_relief')">
                <h3><i class="fas fa-leaf"></i> Stress Relief Technique</h3>
                <p>Quick muscle relaxation for immediate stress reduction.</p>
            </div>
            <div id="meditation-display" style="display: none; margin-top: 30px;"></div>
        </div>

        <div id="challenges-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🏆 Wellness Challenges</h2>
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('challenges')">🎯 Challenges</button>
                <button class="tab-btn" onclick="switchTab('achievements')">⭐ Achievements</button>
            </div>
            <div id="challenges-tab">
                <div id="challenges-list"></div>
            </div>
            <div id="achievements-tab" style="display: none;">
                <div class="badge-container" id="achievements-list"></div>
            </div>
        </div>

        <!-- NEW SCREENS START HERE -->

        <div id="wellness-store-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">💎 Wellness Store</h2>

            <div style="background: var(--glass-bg); padding: 20px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
                <h3>Your Balance: <span id="store-points-balance" style="color: var(--accent); font-size: 1.5rem;">0</span> Points</h3>
            </div>

            <div id="store-items-grid" class="menu-grid"></div>
        </div>

        <div id="buddies-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">👥 Wellness Buddies</h2>

            <div class="form-container" style="max-width: 600px; margin-bottom: 30px;">
                <h3>Connect with a Friend</h3>
                <div class="form-group">
                    <label>Friend's Email</label>
                    <input type="email" id="buddy-email" placeholder="friend@example.com">
                </div>
                <button class="btn" onclick="sendBuddyRequest()">Send Request</button>
            </div>

            <div id="pending-buddy-requests" style="margin-bottom: 30px;"></div>
            <div id="my-buddies-list"></div>
        </div>

        <div id="micro-habits-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🌱 Micro Habit Stacking</h2>

            <div class="insight-box" style="margin-bottom: 30px;">
                <h3>What is Micro Habit Stacking?</h3>
                <p>Tiny habits are 90% more likely to stick when anchored to existing routines. Example: "After I pour my coffee, I will drink one glass of water."</p>
            </div>

            <div class="form-container" style="max-width: 700px; margin-bottom: 30px;">
                <h3>Create a Micro Habit</h3>
                <div class="form-group">
                    <label>What goal do you want to work on?</label>
                    <select id="micro-habit-goal">
                        <option value="water">Hydration</option>
                        <option value="meditation">Meditation</option>
                        <option value="exercise">Exercise</option>
                        <option value="sleep">Better Sleep</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Anchor to this existing habit:</label>
                    <select id="micro-habit-anchor">
                        <option value="Making morning coffee">Making morning coffee</option>
                        <option value="Brushing teeth">Brushing teeth</option>
                        <option value="Taking a shower">Taking a shower</option>
                        <option value="Sitting down at desk">Sitting down at desk</option>
                        <option value="Eating lunch">Eating lunch</option>
                        <option value="Getting into bed">Getting into bed</option>
                    </select>
                </div>
                <button class="btn" onclick="createMicroHabit()">Get Suggestion</button>
            </div>

            <div id="micro-habit-suggestion" style="display: none; margin-bottom: 30px;"></div>
            <div id="my-micro-habits"></div>
        </div>

        <div id="group-challenges-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🏆 Group Challenges</h2>

            <div style="display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap;">
                <button class="btn" onclick="showCreateGroupChallenge()">Create Challenge</button>
                <button class="btn" style="background: var(--glass-bg);" onclick="loadGroupChallenges()">Refresh</button>
            </div>

            <div id="create-group-challenge-form" style="display: none;"></div>
            <div id="group-challenges-list"></div>
        </div>

        <div id="soundscape-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">🎵 Wellness Soundscapes</h2>

            <div class="insight-box" style="margin-bottom: 30px;">
                <h3>Personalized Sonic Healing</h3>
                <p>Choose your current emotion, and I'll generate a 5-minute soundscape with therapeutic frequencies designed to shift your state.</p>
            </div>

            <div class="menu-grid" style="margin-bottom: 30px;">
                <button class="menu-btn" onclick="generateSoundscape('stress')">
                    <i class="fas fa-wind"></i> Feeling Stressed
                </button>
                <button class="menu-btn" onclick="generateSoundscape('anxiety')">
                    <i class="fas fa-heartbeat"></i> Feeling Anxious
                </button>
                <button class="menu-btn" onclick="generateSoundscape('sadness')">
                    <i class="fas fa-cloud-rain"></i> Feeling Sad
                </button>
                <button class="menu-btn" onclick="generateSoundscape('fatigue')">
                    <i class="fas fa-battery-empty"></i> Feeling Tired
                </button>
            </div>

            <div id="soundscape-player" style="display: none;"></div>
        </div>

        <div id="data-export-screen" class="screen">
            <button class="back-btn" onclick="showScreen('main-menu')"><i class="fas fa-arrow-left"></i> Back</button>
            <h2 style="margin-bottom: 30px;">📊 Export Your Data</h2>

            <div class="insight-box" style="margin-bottom: 30px;">
                <h3>Your Data, Your Control</h3>
                <p>Export all your wellness data in standard formats. Share with healthcare providers or keep for your records.</p>
            </div>

            <div class="menu-grid">
                <button class="menu-btn" onclick="exportDataJSON()">
                    <i class="fas fa-file-code"></i>
                    <span>Export as JSON</span>
                    <small style="opacity: 0.7; margin-top: 10px;">Complete data export</small>
                </button>
                <button class="menu-btn" onclick="exportDataCSV()">
                    <i class="fas fa-file-csv"></i>
                    <span>Export as CSV</span>
                    <small style="opacity: 0.7; margin-top: 10px;">Health logs spreadsheet</small>
                </button>
            </div>
        </div>

    </div>

    <script>
        let currentUser = null;
        let authToken = null;
        let charts = {};

        function showScreen(screenId) {
            document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
            document.getElementById(screenId).classList.add('active');

            if (screenId === 'main-menu') updateMainMenu();
            if (screenId === 'dashboard-screen') {
                document.querySelectorAll('.stat-number-enhanced').forEach(el => el.textContent = '...');
                loadDashboard();
            }
            if (screenId === 'goals-screen') loadGoals();
            if (screenId === 'challenges-screen') loadChallenges();
            if (screenId === 'community-screen') loadCommunity();
            if (screenId === 'nutrition-fitness-screen') switchPlanType('diet');
            if (screenId === 'health-journal-screen') switchTrackerType('tracker');

            // NEW: Load data for new screens
            if (screenId === 'wellness-store-screen') loadWellnessStore();
            if (screenId === 'buddies-screen') loadBuddies();
            if (screenId === 'micro-habits-screen') loadMicroHabits();
            if (screenId === 'group-challenges-screen') loadGroupChallenges();
        }

        function showAlert(message, type = 'success') {
            const alert = document.createElement('div');
            alert.style.cssText = `
                position: fixed; top: 20px; right: 20px; padding: 15px 20px;
                background: ${type === 'success' ? 'var(--accent)' : 'var(--error)'};
                color: white; border-radius: 8px; z-index: 2000;
                animation: slideInDown 0.3s ease;
            `;
            alert.textContent = message;
            document.body.appendChild(alert);
            setTimeout(() => alert.remove(), 3000);
        }

        async function loadDailyQuote() {
            try {
                const response = await fetch('/api/quote');
                const data = await response.json();
                document.getElementById('daily-quote-text').textContent = data.quote;
            } catch (e) {
                document.getElementById('daily-quote-text').textContent = "Wellness is a choice, not a destination.";
            }
        }

        async function updateMainMenu() {
            const menuOptions = document.getElementById('menu-options');

            if (currentUser) {
                menuOptions.innerHTML = `
                    <button class="menu-btn" onclick="showScreen('chat-screen')">
                        <i class="fas fa-comments"></i> AI Chat
                    </button>
                    <button class="menu-btn" onclick="showScreen('health-journal-screen')">
                        <i class="fas fa-clipboard-list"></i> Health & Journal
                    </button>
                    <button class="menu-btn" onclick="showScreen('dashboard-screen')">
                        <i class="fas fa-chart-line"></i> Dashboard
                    </button>
                    <button class="menu-btn" onclick="showScreen('goals-screen')">
                        <i class="fas fa-trophy"></i> Goals
                    </button>
                    <button class="menu-btn" onclick="showScreen('nutrition-fitness-screen')">
                        <i class="fas fa-heart-pulse"></i> Nutrition & Fitness
                    </button>
                    <button class="menu-btn" onclick="showScreen('meditation-screen')">
                        <i class="fas fa-spa"></i> Meditation
                    </button>
                    <button class="menu-btn" onclick="showScreen('challenges-screen')">
                        <i class="fas fa-fire"></i> Challenges
                    </button>
                    <button class="menu-btn" onclick="showScreen('community-screen')">
                        <i class="fas fa-users"></i> Community
                    </button>
                    <button class="menu-btn" onclick="showScreen('wellness-store-screen')">
                        <i class="fas fa-store"></i> Wellness Store
                    </button>
                    <button class="menu-btn" onclick="showScreen('buddies-screen')">
                        <i class="fas fa-user-friends"></i> Wellness Buddies
                    </button>
                    <button class="menu-btn" onclick="showScreen('micro-habits-screen')">
                        <i class="fas fa-seedling"></i> Micro Habits
                    </button>
                    <button class="menu-btn" onclick="showScreen('group-challenges-screen')">
                        <i class="fas fa-users-cog"></i> Group Challenges
                    </button>
                    <button class="menu-btn" onclick="showScreen('soundscape-screen')">
                        <i class="fas fa-music"></i> Wellness Vibes
                    </button>
                    <button class="menu-btn" onclick="showScreen('data-export-screen')">
                        <i class="fas fa-download"></i> Export Data
                    </button>
                    <button class="menu-btn" onclick="logout()">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </button>
                `;

                // Load points and nudges
                loadPointsDisplay();
                loadNudgesBanner();
            } else {
                menuOptions.innerHTML = `
                    <button class="menu-btn" onclick="showScreen('register-screen')">
                        <i class="fas fa-user-plus"></i> Register
                    </button>
                    <button class="menu-btn" onclick="showScreen('login-screen')">
                        <i class="fas fa-sign-in-alt"></i> Login
                    </button>
                `;
            }
        }

        // NEW: Points Display
        async function loadPointsDisplay() {
            if (!authToken) return;

            try {
                const response = await fetch('/api/points', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const data = await response.json();

                document.getElementById('points-display').style.display = 'block';
                document.getElementById('points-value').textContent = data.points;
                document.getElementById('user-rank').textContent = data.rank;
            } catch (error) {
                console.error('Failed to load points:', error);
            }
        }

        // NEW: Nudges Banner
        async function loadNudgesBanner() {
            if (!authToken) return;

            try {
                const response = await fetch('/api/nudges', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const data = await response.json();

                if (data.nudges && data.nudges.length > 0) {
                    const nudge = data.nudges[0];
                    const banner = document.getElementById('nudges-banner');

                    const priorityColors = {
                        'high': '#ef5350',
                        'medium': '#ffa726',
                        'low': '#66bb6a'
                    };

                    banner.innerHTML = `
                        <div style="
                            background: linear-gradient(135deg, ${priorityColors[nudge.priority]} 0%, rgba(0,0,0,0.3) 100%);
                            padding: 20px;
                            border-radius: 15px;
                            margin-bottom: 20px;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            animation: slideInDown 0.5s ease;
                        ">
                            <div style="flex: 1;">
                                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">
                                    ${nudge.priority === 'high' ? '⚠️' : nudge.priority === 'medium' ? '💡' : '✨'} Health Insight
                                </div>
                                <div style="margin-bottom: 10px;">${nudge.message}</div>
                                <div style="opacity: 0.8; font-size: 0.9rem;">
                                    <i class="fas fa-lightbulb"></i> Suggested Action: ${nudge.action}
                                </div>
                            </div>
                            <button class="btn" style="width: auto; padding: 10px 20px;" onclick="dismissNudge(${nudge.id})">
                                <i class="fas fa-check"></i> Got it
                            </button>
                        </div>
                    `;
                    banner.style.display = 'block';
                }
            } catch (error) {
                console.error('Failed to load nudges:', error);
            }
        }

        async function dismissNudge(nudgeId) {
            try {
                await fetch(`/api/nudges/${nudgeId}/read`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                document.getElementById('nudges-banner').style.display = 'none';
                setTimeout(loadNudgesBanner, 500);
            } catch (error) {
                console.error('Failed to dismiss nudge:', error);
            }
        }

        setInterval(() => {
            if (authToken) loadNudgesBanner();
        }, 300000);

        document.getElementById('register-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                email: document.getElementById('reg-email').value,
                password: document.getElementById('reg-password').value,
                confirmPassword: document.getElementById('reg-confirm-password').value,
                name: document.getElementById('reg-name').value,
                age: document.getElementById('reg-age').value ? parseInt(document.getElementById('reg-age').value) : null,
                languagePreference: 'English'
            };

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (response.ok) {
                    showAlert('Registration successful! Login now.', 'success');
                    setTimeout(() => showScreen('login-screen'), 1500);
                } else {
                    const result = await response.json();
                    showAlert(result.detail || 'Registration failed', 'error');
                }
            } catch (error) {
                showAlert('Network error during registration', 'error');
            }
        });

        document.getElementById('login-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                email: document.getElementById('login-email').value,
                password: document.getElementById('login-password').value
            };

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                if (response.ok) {
                    currentUser = result.user;
                    authToken = result.token;
                    showAlert(result.message, 'success');
                    showScreen('main-menu');
                    updateMainMenu();
                } else {
                    showAlert(result.detail || 'Login failed', 'error');
                }
            } catch (error) {
                showAlert('Network error during login', 'error');
            }
        });

        function switchPlanType(type) {
            document.getElementById('diet-selector').classList.remove('active');
            document.getElementById('fitness-selector').classList.remove('active');

            if (type === 'diet') {
                document.getElementById('diet-selector').classList.add('active');
            } else {
                document.getElementById('fitness-selector').classList.add('active');
            }

            document.getElementById('diet-plan-section').classList.remove('active');
            document.getElementById('fitness-plan-section').classList.remove('active');

            if (type === 'diet') {
                document.getElementById('diet-plan-section').classList.add('active');
            } else {
                document.getElementById('fitness-plan-section').classList.add('active');
            }
        }

        function switchTrackerType(type) {
            document.getElementById('tracker-selector').classList.remove('active');
            document.getElementById('journal-selector-btn').classList.remove('active');

            if (type === 'tracker') {
                document.getElementById('tracker-selector').classList.add('active');
            } else {
                document.getElementById('journal-selector-btn').classList.add('active');
            }

            document.getElementById('tracker-section').classList.remove('active');
            document.getElementById('journal-section-content').classList.remove('active');

            if (type === 'tracker') {
                document.getElementById('tracker-section').classList.add('active');
                if (authToken) loadHealthLogsDisplay();
            } else {
                document.getElementById('journal-section-content').classList.add('active');
                if (authToken) loadJournalDisplay();
            }
        }

        function usePrompt(promptType) {
            const prompts = {
                'gratitude': "Today I'm grateful for:\\n• ",
                'achievement': "Today I accomplished:\\n• ",
                'challenge': "A challenge I faced today:\\n",
                'lesson': "Today I learned:\\n",
                'goals': "My goals for tomorrow:\\n1. ",
                'reflection': "Reflecting on my day:\\n"
            };

            const contentField = document.getElementById('journal-content');
            const currentContent = contentField.value.trim();

            if (currentContent === '') {
                contentField.value = prompts[promptType];
            } else {
                contentField.value = currentContent + '\\n\\n' + prompts[promptType];
            }

            contentField.focus();
            contentField.setSelectionRange(contentField.value.length, contentField.value.length);
        }

        async function loadHealthLogsDisplay() {
            if (!authToken) {
                document.getElementById('health-logs-display').innerHTML =
                    '<p style="text-align: center; opacity: 0.6;">Please login to view your health logs.</p>';
                return;
            }

            try {
                const response = await fetch('/api/health-logs', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                const logs = await response.json();
                const container = document.getElementById('health-logs-display');

                if (logs.length === 0) {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.6;">No health logs yet. Start tracking!</p>';
                } else {
                    container.innerHTML = logs.slice(0, 10).map(log => `
                        <div class="health-log-card">
                            <h4>
                                <i class="fas fa-calendar-day"></i>
                                ${new Date(log.log_date).toLocaleDateString()}
                            </h4>
                            <div class="health-metrics">
                                ${log.steps ? `<div class="metric-item"><div class="metric-label">Steps</div><div class="metric-value">${log.steps.toLocaleString()}</div></div>` : ''}
                                ${log.calories ? `<div class="metric-item"><div class="metric-label">Calories</div><div class="metric-value">${log.calories}</div></div>` : ''}
                                ${log.sleep_hours ? `<div class="metric-item"><div class="metric-label">Sleep</div><div class="metric-value">${log.sleep_hours}h</div></div>` : ''}
                                ${log.water_intake ? `<div class="metric-item"><div class="metric-label">Water</div><div class="metric-value">${log.water_intake}L</div></div>` : ''}
                                ${log.mood ? `<div class="metric-item"><div class="metric-label">Mood</div><div class="metric-value">${log.mood}</div></div>` : ''}
                            </div>
                            ${log.notes ? `<div class="log-notes"><i class="fas fa-comment-dots"></i> ${log.notes}</div>` : ''}
                        </div>
                    `).join('');
                }
            } catch (error) {
                document.getElementById('health-logs-display').innerHTML =
                    '<p style="text-align: center; color: var(--error);">Failed to load health logs.</p>';
            }
        }

        async function loadJournalDisplay() {
            if (!authToken) {
                document.getElementById('journal-entries-display').innerHTML =
                    '<p style="text-align: center; opacity: 0.6;">Please login to view your journal entries.</p>';
                return;
            }

            try {
                const response = await fetch('/api/journal', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                const entries = await response.json();
                const container = document.getElementById('journal-entries-display');

                if (entries.length === 0) {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.6;">No journal entries yet. Start writing!</p>';
                } else {
                    container.innerHTML = entries.slice(0, 10).map(e => `
                        <div class="journal-entry">
                            <h4>${e.title}</h4>
                            <small>${new Date(e.created_at).toLocaleDateString()}</small>
                            <p>${e.content}</p>
                        </div>
                    `).join('');
                }
            } catch (error) {
                document.getElementById('journal-entries-display').innerHTML =
                    '<p style="text-align: center; color: var(--error);">Failed to load journal entries.</p>';
            }
        }

        function addMessage(text, isUser = false) {
            const container = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            messageDiv.innerHTML = `
                <div class="message-avatar"><i class="fas fa-${isUser ? 'user' : 'robot'}"></i></div>
                <div class="message-content">${text}</div>
            `;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;

            addMessage(message, true);
            input.value = '';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': authToken ? `Bearer ${authToken}` : ''
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                if (response.ok) {
                    addMessage(data.bot_response, false);
                } else {
                    addMessage('Sorry, I encountered an error.', false);
                }
            } catch (error) {
                addMessage('Network error. Please try again.', false);
            }
        }

        function handleChatKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }

        document.getElementById('health-log-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            const logData = {
                steps: document.getElementById('log-steps').value ? parseInt(document.getElementById('log-steps').value) : null,
                calories: document.getElementById('log-calories').value ? parseInt(document.getElementById('log-calories').value) : null,
                sleep_hours: document.getElementById('log-sleep').value ? parseFloat(document.getElementById('log-sleep').value) : null,
                water_intake: document.getElementById('log-water').value ? parseFloat(document.getElementById('log-water').value) : null,
                mood: document.getElementById('log-mood').value || null,
                notes: document.getElementById('log-notes').value || null
            };

            try {
                const response = await fetch('/api/health-log', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify(logData)
                });

                const result = await response.json();
                if (response.ok) {
                    showAlert(result.message, 'success');
                    this.reset();
                    loadHealthLogsDisplay();
                    loadPointsDisplay();
                } else {
                    showAlert(result.detail || 'Failed to save', 'error');
                }
            } catch (error) {
                showAlert('Network error', 'error');
            }
        });

        document.getElementById('goal-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            const data = {
                goal_type: document.getElementById('goal-type').value,
                target_value: parseFloat(document.getElementById('goal-target').value),
                description: document.getElementById('goal-description').value
            };

            try {
                const response = await fetch('/api/goals', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                if (response.ok) {
                    showAlert('Goal created successfully!', 'success');
                    this.reset();
                    loadGoals();
                } else {
                    showAlert(result.message || 'Failed to create goal', 'error');
                }
            } catch (error) {
                showAlert('Failed to create goal', 'error');
            }
        });

        async function loadGoals() {
            if (!authToken) return;

            try {
                const response = await fetch('/api/goals', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const goals = await response.json();

                const container = document.getElementById('goals-list');
                if (goals.length === 0) {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.6;">No goals yet. Create your first goal!</p>';
                } else {
                    container.innerHTML = goals.map(g => `
                        <div class="goal-card">
                            ${g.current_streak >= 7 ? `<div class="streak-badge">🔥 ${g.current_streak} days</div>` : ''}
                            <h4>${g.goal_type.charAt(0).toUpperCase() + g.goal_type.slice(1)}</h4>
                            <p>${g.description || 'Daily goal'}</p>
                            <p>Target: ${g.target_value} | Current: ${g.current_value}</p>
                            <div class="goal-progress">
                                <div class="goal-progress-bar" style="width: ${Math.min((g.current_value / g.target_value) * 100, 100)}%"></div>
                            </div>
                            <button class="btn" style="margin-top: 15px; width: 100%;" onclick="updateGoalProgress(${g.id}, ${g.target_value})">
                                Mark Today Complete
                            </button>
                        </div>
                    `).join('');
                }
            } catch (error) {
                showAlert('Failed to load goals', 'error');
            }
        }

        async function updateGoalProgress(goalId, targetValue) {
            try {
                const response = await fetch(`/api/goals/${goalId}/progress`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ value: targetValue })
                });

                const result = await response.json();
                if (response.ok && result.success) {
                    showAlert(result.message, 'success');
                    loadGoals();
                    loadPointsDisplay();
                } else {
                    showAlert(result.message || 'Failed to update progress', 'error');
                }
            } catch (error) {
                showAlert('Network error while updating progress', 'error');
            }
        }

        // NEW: Enhanced Diet Plan with LLM
        document.getElementById('diet-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            const data = {
                diet_preference: document.getElementById('diet-preference').value,
                health_goal: document.getElementById('health-goal').value,
                calories_target: parseInt(document.getElementById('calories-target').value),
                allergies: ''
            };

            try {
                const response = await fetch('/api/llm/diet-plan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify(data)
                });

                const plan = await response.json();
                displayEnhancedDietPlan(plan);
                showAlert('AI-powered diet plan generated! 🤖', 'success');
            } catch (error) {
                showAlert('Failed to generate diet plan', 'error');
            }
        });

        function displayEnhancedDietPlan(plan) {
            const container = document.getElementById('diet-plan-display');
            container.innerHTML = `
                <div class="health-tracker">
                    <div style="background: linear-gradient(135deg, rgba(0,201,167,0.2) 0%, rgba(66,133,244,0.2) 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 2px solid var(--accent);">
                        <h3 style="color: var(--accent); margin-bottom: 15px;">🤖 AI-Generated Personalized Plan</h3>
                        <p style="line-height: 1.6; opacity: 0.9;">${plan.reasoning || 'Your personalized plan based on health data'}</p>
                        ${plan.personalization_note ? `<p style="margin-top: 10px; opacity: 0.8; font-size: 0.9rem;">📊 ${plan.personalization_note}</p>` : ''}
                    </div>

                    ${plan.nutrition ? `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 25px;">
                        <div style="background: rgba(255,255,255,0.08); padding: 15px; border-radius: 12px; text-align: center;">
                            <div style="opacity: 0.7; font-size: 0.8rem; margin-bottom: 5px;">Calories</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent);">${plan.nutrition.daily_calories}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); padding: 15px; border-radius: 12px; text-align: center;">
                            <div style="opacity: 0.7; font-size: 0.8rem; margin-bottom: 5px;">Protein</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent);">${plan.nutrition.protein_g}g</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); padding: 15px; border-radius: 12px; text-align: center;">
                            <div style="opacity: 0.7; font-size: 0.8rem; margin-bottom: 5px;">Carbs</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent);">${plan.nutrition.carbs_g}g</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); padding: 15px; border-radius: 12px; text-align: center;">
                            <div style="opacity: 0.7; font-size: 0.8rem; margin-bottom: 5px;">Fats</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent);">${plan.nutrition.fats_g}g</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); padding: 15px; border-radius: 12px; text-align: center;">
                            <div style="opacity: 0.7; font-size: 0.8rem; margin-bottom: 5px;">Water</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent);">${plan.nutrition.water_liters}L</div>
                        </div>
                    </div>
                    ` : ''}

                    <div style="margin-top: 20px;">
                        <h4 style="color: var(--accent); margin: 15px 0 10px 0;">🌅 Breakfast</h4>
                        ${plan.meals.breakfast.map(m => `<div style="padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 8px; border-left: 3px solid var(--accent);">${m}</div>`).join('')}

                        <h4 style="color: var(--accent); margin: 15px 0 10px 0;">🍱 Lunch</h4>
                        ${plan.meals.lunch.map(m => `<div style="padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 8px; border-left: 3px solid var(--accent);">${m}</div>`).join('')}

                        <h4 style="color: var(--accent); margin: 15px 0 10px 0;">🍽️ Dinner</h4>
                        ${plan.meals.dinner.map(m => `<div style="padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 8px; border-left: 3px solid var(--accent);">${m}</div>`).join('')}

                        <h4 style="color: var(--accent); margin: 15px 0 10px 0;">🍎 Snacks</h4>
                        ${plan.meals.snacks.map(m => `<div style="padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 8px; border-left: 3px solid var(--accent);">${m}</div>`).join('')}
                    </div>

                    ${plan.local_alternatives ? `
                        <div style="margin-top: 25px; background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                            <h4 style="color: var(--accent); margin-bottom: 15px;">🇮🇳 Local Indian Alternatives</h4>

                            <div style="margin-bottom: 15px;">
                                <strong style="opacity: 0.8;">Breakfast:</strong>
                                <div style="margin-top: 8px;">${plan.local_alternatives.breakfast.join(' • ')}</div>
                            </div>

                            <div style="margin-bottom: 15px;">
                                <strong style="opacity: 0.8;">Lunch:</strong>
                                <div style="margin-top: 8px;">${plan.local_alternatives.lunch.join(' • ')}</div>
                            </div>

                            <div style="margin-bottom: 15px;">
                                <strong style="opacity: 0.8;">Dinner:</strong>
                                <div style="margin-top: 8px;">${plan.local_alternatives.dinner.join(' • ')}</div>
                            </div>

                            <div>
                                <strong style="opacity: 0.8;">Snacks:</strong>
                                <div style="margin-top: 8px;">${plan.local_alternatives.snacks.join(' • ')}</div>
                            </div>
                        </div>
                    ` : ''}

                    ${plan.tips && plan.tips.length > 0 ? `
                        <div style="margin-top: 25px; background: linear-gradient(135deg, rgba(0,201,167,0.1) 0%, rgba(66,133,244,0.1) 100%); padding: 20px; border-radius: 15px; border: 1px solid var(--accent);">
                            <h4 style="color: var(--accent); margin-bottom: 15px;">💡 Personalized Tips for You</h4>
                            <ul style="list-style: none; padding: 0;">
                                ${plan.tips.map(tip => `
                                    <li style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                        ${tip}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        // NEW: Enhanced Fitness Plan with LLM
        document.getElementById('fitness-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            const data = {
                fitness_level: document.getElementById('fitness-level').value,
                goal: document.getElementById('fitness-goal').value
            };

            try {
                const response = await fetch('/api/llm/fitness-plan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify(data)
                });

                const plan = await response.json();
                displayEnhancedFitnessPlan(plan);
                showAlert('AI-powered fitness plan generated! 💪', 'success');
            } catch (error) {
                showAlert('Failed to generate fitness plan', 'error');
            }
        });

        function displayEnhancedFitnessPlan(plan) {
            const container = document.getElementById('fitness-plan-display');
            const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];

            container.innerHTML = `
                <div class="health-tracker">
                    <div style="background: linear-gradient(135deg, rgba(0,201,167,0.2) 0%, rgba(66,133,244,0.2) 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 2px solid var(--accent);">
                        <h3 style="color: var(--accent); margin-bottom: 15px;">🤖 AI-Generated Fitness Plan</h3>
                        <p style="line-height: 1.6; opacity: 0.9;">${plan.reasoning || 'Your personalized workout plan'}</p>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px;">
                        ${days.map((day, index) => `
                            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border-left: 3px solid ${index === 2 || index === 6 ? '#ffa726' : 'var(--accent)'};">
                                <h4 style="color: ${index === 2 || index === 6 ? '#ffa726' : 'var(--accent)'}; margin-bottom: 10px; text-transform: capitalize;">
                                    ${day} ${index === 2 || index === 6 ? '🌟' : ''}
                                </h4>
                                <p style="font-size: 0.9rem; line-height: 1.5;">${plan.weekly_plan[day]}</p>
                            </div>
                        `).join('')}
                    </div>

                    ${plan.progression ? `
                        <div style="margin-top: 25px; background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">
                            <h4 style="color: var(--accent); margin-bottom: 15px;">📈 Progression Plan</h4>
                            <ul style="list-style: none; padding: 0;">
                                ${plan.progression.map(step => `
                                    <li style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                        ✓ ${step}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}

                    ${plan.recovery_tips ? `
                        <div style="margin-top: 25px; background: linear-gradient(135deg, rgba(255,167,38,0.1) 0%, rgba(239,83,80,0.1) 100%); padding: 20px; border-radius: 15px; border: 1px solid #ffa726;">
                            <h4 style="color: #ffa726; margin-bottom: 15px;">🔄 Recovery Tips</h4>
                            <ul style="list-style: none; padding: 0;">
                                ${plan.recovery_tips.map(tip => `
                                    <li style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                        ${tip}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        // NEW: CBT-Enhanced Journal
        document.getElementById('journal-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            const entry = {
                title: document.getElementById('journal-title').value,
                content: document.getElementById('journal-content').value
            };

            try {
                const response = await fetch('/api/journal', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify(entry)
                });

                if (response.ok) {
                    const result = await response.json();
                    const entryId = result.entry_id;

                    // Analyze for CBT patterns
                    const cbtResponse = await fetch('/api/journal/cbt-analysis', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${authToken}`
                        },
                        body: JSON.stringify({
                            content: entry.content,
                            entry_id: entryId
                        })
                    });

                    const cbtResult = await cbtResponse.json();

                    if (cbtResult.patterns_detected && cbtResult.patterns_detected.length > 0) {
                        showCBTFeedback(cbtResult.patterns_detected, cbtResult.cbt_guidance);
                    } else {
                        showAlert('Journal entry saved! 📝', 'success');
                    }

                    this.reset();
                    loadJournalDisplay();
                    loadPointsDisplay();
                }
            } catch (error) {
                showAlert('Failed to save entry', 'error');
            }
        });

        function showCBTFeedback(patterns, guidance) {
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.85); display: flex; align-items: center;
                justify-content: center; z-index: 2000; padding: 20px; overflow-y: auto;
            `;

            modal.innerHTML = `
                <div style="background: var(--bg-light); padding: 40px; border-radius: 25px; max-width: 700px; max-height: 90vh; overflow-y: auto; position: relative;">
                    <button onclick="this.parentElement.parentElement.remove()" style="position: absolute; top: 15px; right: 15px; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer;">×</button>

                    <div style="text-align: center; margin-bottom: 30px;">
                        <div style="font-size: 3rem; margin-bottom: 15px;">🧠</div>
                        <h2>CBT Thought Pattern Analysis</h2>
                        <p style="opacity: 0.8; margin-top: 10px;">${guidance}</p>
                    </div>

                    ${patterns.map((pattern, index) => `
                        <div style="background: rgba(255,255,255,0.08); padding: 25px; border-radius: 15px; margin-bottom: 20px; border-left: 4px solid var(--accent);">
                            <h4 style="color: var(--accent); margin-bottom: 15px; text-transform: capitalize;">
                                Pattern ${index + 1}: ${pattern.pattern.replace(/_/g, ' ')}
                            </h4>

                            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                                <strong style="opacity: 0.8;">Reframe Prompt:</strong>
                                <p style="margin-top: 8px; line-height: 1.6;">${pattern.reframe_prompt}</p>
                            </div>

                            <div style="background: rgba(0,201,167,0.1); padding: 15px; border-radius: 10px; border: 1px solid var(--accent);">
                                <strong style="opacity: 0.8;">Example:</strong>
                                <p style="margin-top: 8px; line-height: 1.6; font-style: italic;">${pattern.example}</p>
                            </div>
                        </div>
                    `).join('')}

                    <div style="background: linear-gradient(135deg, rgba(0,201,167,0.15) 0%, rgba(66,133,244,0.15) 100%); padding: 20px; border-radius: 15px; margin-top: 25px; border: 2px solid var(--accent);">
                        <h4 style="margin-bottom: 10px;">💡 Next Steps</h4>
                        <p style="line-height: 1.6;">
                            Try rewriting your thoughts using the reframe prompts above. Remember: thoughts aren't facts,
                            they're just thoughts. You have the power to choose how you think about things.
                        </p>
                    </div>

                    <button class="btn" style="width: 100%; margin-top: 25px;" onclick="this.parentElement.parentElement.remove()">
                        Thank you, I understand
                    </button>
                </div>
            `;

            document.body.appendChild(modal);
        }

        document.getElementById('community-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            const data = {
                title: document.getElementById('post-title').value,
                content: document.getElementById('post-content').value,
                anonymous: document.getElementById('post-anonymous').checked
            };

            try {
                const response = await fetch('/api/community/posts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
                    body: JSON.stringify(data)
                });

                if (response.ok) {
                    showAlert('Post shared successfully!', 'success');
                    this.reset();
                    loadCommunity();
                }
            } catch (error) {
                showAlert('Failed to create post', 'error');
            }
        });

        async function loadCommunity() {
            try {
                const response = await fetch('/api/community/posts');
                const posts = await response.json();

                const container = document.getElementById('community-posts');
                if (posts.length === 0) {
                    container.innerHTML = '<p style="text-align: center; opacity:0.6;">No posts yet. Be the first to share!</p>';
                } else {
                    container.innerHTML = posts.map(p => `
                        <div style="background: rgba(255,255,255,0.06); padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 4px solid var(--accent);">
                            <h3>${p.title}</h3>
                            <p style="opacity: 0.7; font-size: 0.9rem; margin: 10px 0;">By ${p.author}</p>
                            <p>${p.content}</p>
                        </div>
                    `).join('');
                }
            } catch (error) {
                showAlert('Failed to load posts', 'error');
            }
        }

        async function startMeditation(type) {
            try {
                const response = await fetch(`/api/meditation/${type}`);
                const data = await response.json();

                const display = document.getElementById('meditation-display');
                display.style.display = 'block';
                display.innerHTML = `
                    <div style="background: linear-gradient(135deg, rgba(0, 201, 167, 0.2) 0%, rgba(66, 133, 244, 0.2) 100%); border: 2px solid var(--accent); padding: 30px; border-radius: 15px;">
                        <pre style="white-space: pre-wrap; line-height: 1.8; font-family: 'Inter'; color: white;">${data.meditation}</pre>
                        <button class="btn" style="margin-top: 20px;" onclick="document.getElementById('meditation-display').style.display='none';">Close</button>
                    </div>
                `;
                display.scrollIntoView({ behavior: 'smooth' });
            } catch (error) {
                showAlert('Failed to load meditation', 'error');
            }
        }

        async function loadChallenges() {
            if (!authToken) { showAlert('Please login first', 'error'); return; }

            try {
                const response = await fetch('/api/challenges');
                const challenges = await response.json();

                const container = document.getElementById('challenges-list');
                container.innerHTML = challenges.map(c => `
                    <div class="challenge-card">
                        <div>
                            <h3>${c.name}</h3>
                            <p>${c.description}</p>
                            <small>Difficulty: ${c.difficulty} | Reward: ${c.reward} pts</small>
                        </div>
                        <button class="btn" style="width: auto; padding: 10px 20px;" onclick="completeChallenge('${c.name}')">Complete</button>
                    </div>
                `).join('');

                const achResponse = await fetch('/api/achievements', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const achievements = await achResponse.json();

                const achContainer = document.getElementById('achievements-list');
                if (achievements.length === 0) {
                    achContainer.innerHTML = '<p style="text-align: center; opacity: 0.6; grid-column: 1/-1;">Complete challenges to earn badges!</p>';
                } else {
                    achContainer.innerHTML = achievements.map(a => `
                        <div class="badge">
                            <div style="font-size: 3rem; margin-bottom: 10px;">${a.icon}</div>
                            <div style="font-size: 0.8rem; font-weight: 600;">${a.name}</div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error(error);
            }
        }

        async function completeChallenge(name) {
            try {
                const response = await fetch('/api/complete-challenge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ challenge_name: name })
                });

                if (response.ok) {
                    showAlert('🎉 Challenge completed!', 'success');
                    loadChallenges();
                    loadPointsDisplay();
                }
            } catch (error) {
                console.error(error);
            }
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('[id$="-tab"]').forEach(t => t.style.display = 'none');

            event.target.classList.add('active');
            document.getElementById(tab + '-tab').style.display = 'block';
        }

        async function loadDashboard() {
            if (!authToken) {
                showAlert('Please login first', 'error');
                return;
            }

            try {
                const response = await fetch('/api/analytics', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                if (!response.ok) {
                    throw new Error('Failed to fetch analytics');
                }

                const data = await response.json();

                animateCounter('enhanced-chats', data.total_chats || 0);
                animateCounter('enhanced-health-logs', data.health_logs_count || 0);
                animateCounter('enhanced-goals', data.active_goals || 0);
                animateCounter('enhanced-streak', data.total_streak || 0);

                createEnhancedSentimentChart(data.sentiment_distribution || {});
                createEnhancedEmotionChart(data.emotion_distribution || {});
                createActivityTimelineChart(data.recent_activity || {});

                const logsResponse = await fetch('/api/health-logs', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const logs = await logsResponse.json();

                generatePersonalizedInsights(logs, data);

            } catch (error) {
                console.error('Error loading dashboard:', error);
                showAlert('Failed to load dashboard data', 'error');
            }
        }

        function animateCounter(elementId, targetValue) {
            const element = document.getElementById(elementId);
            if (!element) return;

            const duration = 2000;
            const start = 0;
            const increment = targetValue / (duration / 16);
            let current = start;

            const timer = setInterval(() => {
                current += increment;
                if (current >= targetValue) {
                    element.textContent = Math.round(targetValue);
                    clearInterval(timer);
                } else {
                    element.textContent = Math.round(current);
                }
            }, 16);
        }

        function createEnhancedSentimentChart(data) {
            const ctx = document.getElementById('enhanced-sentiment-chart');
            if (!ctx) return;

            const context = ctx.getContext('2d');

            if (window.enhancedSentimentChart) {
                window.enhancedSentimentChart.destroy();
            }

            const labels = Object.keys(data).length > 0
                ? Object.keys(data).map(k => k.charAt(0).toUpperCase() + k.slice(1))
                : ['Positive', 'Neutral', 'Negative'];

            const values = Object.keys(data).length > 0
                ? Object.values(data)
                : [0, 0, 0];

            window.enhancedSentimentChart = new Chart(context, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: [
                            'rgba(76, 175, 80, 0.8)',
                            'rgba(255, 193, 7, 0.8)',
                            'rgba(244, 67, 54, 0.8)'
                        ],
                        borderColor: [
                            'rgb(76, 175, 80)',
                            'rgb(255, 193, 7)',
                            'rgb(244, 67, 54)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: 'white',
                                font: { size: 14, weight: 'bold' },
                                padding: 15
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            titleFont: { size: 16 },
                            bodyFont: { size: 14 },
                            callbacks: {
                                label: function(context) {
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = total > 0
                                        ? ((context.parsed * 100) / total).toFixed(1)
                                        : 0;
                                    return `${context.label}: ${context.parsed} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
        }

        function createEnhancedEmotionChart(data) {
            const ctx = document.getElementById('enhanced-emotion-chart');
            if (!ctx) return;

            const context = ctx.getContext('2d');

            if (window.enhancedEmotionChart) {
                window.enhancedEmotionChart.destroy();
            }

            const labels = Object.keys(data).length > 0
                ? Object.keys(data).map(k => k.charAt(0).toUpperCase() + k.slice(1))
                : ['Joy', 'Stress', 'Anxiety', 'Sadness', 'Fatigue', 'Anger'];

            const values = Object.keys(data).length > 0
                ? Object.values(data)
                : [0, 0, 0, 0, 0, 0];

            window.enhancedEmotionChart = new Chart(context, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Frequency',
                        data: values,
                        backgroundColor: 'rgba(0, 201, 167, 0.8)',
                        borderColor: 'rgb(0, 201, 167)',
                        borderWidth: 2,
                        borderRadius: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true,
                            ticks: { color: 'white', font: { size: 12, weight: 'bold' } },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: 'white', font: { size: 12, weight: 'bold' } },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12
                        }
                    }
                }
            });
        }

        function createActivityTimelineChart(data) {
            const ctx = document.getElementById('activity-timeline-chart');
            if (!ctx) return;

            const context = ctx.getContext('2d');

            if (window.activityTimelineChart) {
                window.activityTimelineChart.destroy();
            }

            const dates = Object.keys(data).length > 0
                ? Object.keys(data)
                : ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];

            const values = Object.keys(data).length > 0
                ? Object.values(data)
                : [0, 0, 0, 0, 0, 0, 0];

            window.activityTimelineChart = new Chart(context, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: 'Daily Activity',
                        data: values,
                        borderColor: 'rgb(0, 201, 167)',
                        backgroundColor: 'rgba(0, 201, 167, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 6,
                        pointBackgroundColor: 'rgb(0, 201, 167)',
                        pointBorderColor: 'white',
                        pointBorderWidth: 2,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: 'white', font: { size: 12, weight: 'bold' } },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                            ticks: { color: 'white', font: { size: 12, weight: 'bold' } },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            callbacks: {
                                label: function(context) {
                                    return `Activity: ${context.parsed.y} interactions`;
                                }
                            }
                        }
                    }
                }
            });
        }

        function generatePersonalizedInsights(logs, analyticsData) {
            if (!logs || logs.length === 0) {
                return;
            }

            const insights = {
                healthScore: calculateHealthScore(logs),
                topStrength: identifyTopStrength(logs),
                improvementArea: identifyImprovementArea(logs),
                avgSleep: calculateAverage(logs, 'sleep_hours'),
                avgSteps: calculateAverage(logs, 'steps'),
                avgWater: calculateAverage(logs, 'water_intake'),
                moodTrend: getMoodTrend(logs),
                recommendations: generateRecommendations(logs, analyticsData)
            };

            updateInsightCard(insights);
            updateProgressBars(logs);
            updateQuickStats(logs);
            displayRecommendations(insights.recommendations);
        }

        function calculateHealthScore(logs) {
            if (!logs || logs.length === 0) return 0;

            let score = 50;

            const avgSleep = logs.reduce((sum, log) => sum + (log.sleep_hours || 0), 0) / logs.length;
            if (avgSleep >= 7 && avgSleep <= 9) score += 20;
            else if (avgSleep >= 6 && avgSleep < 10) score += 15;

            const avgWater = logs.reduce((sum, log) => sum + (log.water_intake || 0), 0) / logs.length;
            if (avgWater >= 2.5) score += 15;
            else if (avgWater >= 2) score += 10;

            const avgSteps = logs.reduce((sum, log) => sum + (log.steps || 0), 0) / logs.length;
            if (avgSteps >= 8000) score += 15;
            else if (avgSteps >= 5000) score += 10;

            if (logs.length >= 7) score += 10;
            else if (logs.length >= 3) score += 5;

            return Math.min(100, score);
        }

        function identifyTopStrength(logs) {
            if (!logs || logs.length === 0) return 'No data yet';

            const metrics = {
                sleep: logs.filter(l => l.sleep_hours && l.sleep_hours >= 7).length,
                water: logs.filter(l => l.water_intake && l.water_intake >= 2.5).length,
                steps: logs.filter(l => l.steps && l.steps >= 8000).length,
                consistency: logs.length
            };

            const topMetric = Object.entries(metrics).reduce((a, b) => b[1] > a[1] ? b : a);

            const strengthMap = {
                sleep: `⚡ Sleep Consistency - ${topMetric[1]} days with 7+ hours`,
                water: `💧 Hydration - ${topMetric[1]} days meeting water goals`,
                steps: `🚶 Activity Level - ${topMetric[1]} days with 8000+ steps`,
                consistency: `📈 Commitment - ${topMetric[1]} days of tracking`
            };

            return strengthMap[topMetric[0]];
        }

        function identifyImprovementArea(logs) {
            if (!logs || logs.length === 0) return 'Start logging to get recommendations';

            const metrics = {
                sleep: logs.filter(l => !l.sleep_hours || l.sleep_hours < 7).length,
                water: logs.filter(l => !l.water_intake || l.water_intake < 2.5).length,
                steps: logs.filter(l => !l.steps || l.steps < 5000).length
            };

            const improvement = Object.entries(metrics).reduce((a, b) => b[1] > a[1] ? b : a);

            const improvementMap = {
                sleep: `😴 Sleep Quality - Average: ${(logs.reduce((sum, l) => sum + (l.sleep_hours || 0), 0) / logs.length).toFixed(1)} hrs. Aim for 7-9 hours`,
                water: `💧 Hydration - Average: ${(logs.reduce((sum, l) => sum + (l.water_intake || 0), 0) / logs.length).toFixed(1)}L. Target: 3-4L daily`,
                steps: `🚶 Activity - Average: ${Math.round(logs.reduce((sum, l) => sum + (l.steps || 0), 0) / logs.length)} steps. Goal: 10,000 steps`
            };

            return improvementMap[improvement[0]];
        }

        function calculateAverage(logs, field) {
            if (!logs || logs.length === 0) return 0;
            const sum = logs.reduce((acc, log) => acc + (log[field] || 0), 0);
            return (sum / logs.length).toFixed(1);
        }

        function getMoodTrend(logs) {
            if (!logs || logs.length === 0) return 'No mood data';

            const recentMoods = logs.slice(0, 7).map(l => l.mood).filter(m => m);

            if (recentMoods.length === 0) return 'Start logging mood to see trends';

            const moodCounts = recentMoods.reduce((acc, mood) => {
                acc[mood] = (acc[mood] || 0) + 1;
                return acc;
            }, {});

            const dominantMood = Object.entries(moodCounts).reduce((a, b) => b[1] > a[1] ? b : a)[0];
            const moodEmoji = {
                'excellent': '😄',
                'good': '😊',
                'okay': '😐',
                'bad': '😔'
            };

            return `${moodEmoji[dominantMood] || '😊'} Trending: ${dominantMood || 'neutral'} mood`;
        }

        function generateRecommendations(logs, analyticsData) {
            const recommendations = [];

            const avgSleep = calculateAverage(logs, 'sleep_hours');
            if (avgSleep < 7) {
                recommendations.push('🌙 Try to get 7-9 hours of sleep. Use our meditation for better sleep quality.');
            }

            const avgWater = calculateAverage(logs, 'water_intake');
            if (avgWater < 2.5) {
                recommendations.push('💧 Increase water intake to 3-4 liters daily for better hydration.');
            }

            const avgSteps = calculateAverage(logs, 'steps');
            if (avgSteps < 8000) {
                recommendations.push('🚶 Aim for 10,000 steps daily. Try our fitness plan for personalized workouts.');
            }

            if (analyticsData.emotion_distribution && analyticsData.emotion_distribution.stress > 0) {
                recommendations.push('🧘 Try our meditation exercises to manage stress and anxiety.');
            }

            return recommendations.slice(0, 3);
        }

        function updateInsightCard(insights) {
            const healthScoreElement = document.getElementById('health-score-insight');
            if (healthScoreElement) {
                healthScoreElement.innerHTML = `
                    Your overall wellness score is <strong>${insights.healthScore}/100</strong>.
                    ${insights.healthScore >= 80 ? '🌟 Excellent work!' :
                      insights.healthScore >= 60 ? '✨ Keep going!' :
                      '💪 You can improve!'}
                `;
            }

            const topStrengthElement = document.getElementById('top-strength-insight');
            if (topStrengthElement) {
                topStrengthElement.innerHTML = insights.topStrength;
            }

            const improvementElement = document.getElementById('improvement-area-insight');
            if (improvementElement) {
                improvementElement.innerHTML = insights.improvementArea;
            }
        }

        function updateProgressBars(logs) {
            if (!logs || logs.length === 0) {
                document.getElementById('no-data-message').style.display = 'block';
                return;
            }

            document.getElementById('no-data-message').style.display = 'none';

            const avgWater = calculateAverage(logs, 'water_intake');
            const avgSteps = calculateAverage(logs, 'steps');
            const avgSleep = calculateAverage(logs, 'sleep_hours');

            const waterPercent = Math.min((avgWater / 3) * 100, 100);
            updateProgressBar('water-progress-bar', 'water-progress-percent', waterPercent);

            const exercisePercent = Math.min((avgSteps / 10000) * 100, 100);
            updateProgressBar('exercise-progress-bar', 'exercise-progress-percent', exercisePercent);

            const sleepPercent = avgSleep >= 9 ? 100 : Math.min((avgSleep / 9) * 100, 100);
            updateProgressBar('sleep-progress-bar', 'sleep-progress-percent', sleepPercent);

            const stepsPercent = Math.min((avgSteps / 10000) * 100, 100);
            updateProgressBar('steps-progress-bar', 'steps-progress-percent', stepsPercent);
        }

        function updateProgressBar(barId, percentId, percentage) {
            const bar = document.getElementById(barId);
            const percent = document.getElementById(percentId);

            if (bar) {
                bar.style.width = percentage + '%';
            }

            if (percent) {
                percent.textContent = Math.round(percentage) + '%';
            }
        }

        function updateQuickStats(logs) {
            if (!logs || logs.length === 0) return;

            const avgSleep = calculateAverage(logs, 'sleep_hours');
            const sleepElement = document.getElementById('avg-sleep-stat');
            if (sleepElement) {
                sleepElement.textContent = avgSleep !== '0' ? avgSleep + ' hrs' : '-- hrs';
            }

            const avgSteps = Math.round(calculateAverage(logs, 'steps'));
            const stepsElement = document.getElementById('avg-steps-stat');
            if (stepsElement) {
                stepsElement.textContent = avgSteps !== 0 ? avgSteps.toLocaleString() + ' steps' : '-- steps';
            }

            const avgWater = calculateAverage(logs, 'water_intake');
            const waterElement = document.getElementById('avg-water-stat');
            if (waterElement) {
                waterElement.textContent = avgWater !== '0' ? avgWater + ' L' : '-- L';
            }

            const moodTrend = getMoodTrend(logs);
            const moodElement = document.getElementById('mood-trend-stat');
            const moodTextElement = document.getElementById('mood-trend-text');

            if (moodElement && moodTextElement) {
                const moodEmoji = moodTrend.split(' ')[0];
                const moodText = moodTrend.split('Trending: ')[1] || 'No mood data';
                moodElement.textContent = moodEmoji;
                moodTextElement.textContent = moodText;
            }
        }

        function displayRecommendations(recommendations) {
            const container = document.getElementById('recommendations-list');

            if (!recommendations || recommendations.length === 0) {
                container.innerHTML = '<p style="opacity: 0.7; text-align: center;">Keep logging to receive personalized recommendations!</p>';
                return;
            }

            container.innerHTML = `
                <ul style="list-style: none; padding: 0;">
                    ${recommendations.map((rec, index) => `
                        <li style="
                            display: flex;
                            align-items: center;
                            padding: 15px;
                            background: rgba(0, 201, 167, 0.1);
                            border-left: 4px solid var(--accent);
                            margin-bottom: 12px;
                            border-radius: 8px;
                            animation: slideIn 0.3s ease ${index * 0.1}s both;
                        ">
                            <span style="margin-right: 15px; font-size: 1.2rem;">${rec.split(' ')[0]}</span>
                            <span style="flex: 1;">${rec}</span>
                        </li>
                    `).join('')}
                </ul>
                <style>
                    @keyframes slideIn {
                        from {
                            opacity: 0;
                            transform: translateX(-20px);
                        }
                        to {
                            opacity: 1;
                            transform: translateX(0);
                        }
                    }
                </style>
            `;
        }





        // NEW FEATURES JAVASCRIPT

        async function loadWellnessStore() {
            if (!authToken) return;

            try {
                const pointsResponse = await fetch('/api/points', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const pointsData = await pointsResponse.json();
                document.getElementById('store-points-balance').textContent = pointsData.points;

                const storeResponse = await fetch('/api/store');
                const storeData = await storeResponse.json();

                const grid = document.getElementById('store-items-grid');
                grid.innerHTML = storeData.items.map(item => `
                    <div class="stat-card" style="text-align: left; position: relative;">
                        <div style="position: absolute; top: 15px; right: 15px; background: var(--accent); padding: 5px 12px; border-radius: 20px; font-weight: 700;">
                            ${item.cost_points} pts
                        </div>
                        <div style="font-size: 2rem; margin-bottom: 10px;">
                            ${getCategoryIcon(item.category)}
                        </div>
                        <h3 style="margin-bottom: 10px;">${item.name}</h3>
                        <p style="opacity: 0.7; font-size: 0.9rem; margin-bottom: 15px;">${item.description}</p>
                        <button class="btn" style="width: 100%; padding: 10px;"
                            onclick="purchaseItem(${item.id}, ${item.cost_points}, '${item.name}')"
                            ${pointsData.points < item.cost_points ? 'disabled' : ''}>
                            ${pointsData.points >= item.cost_points ? 'Purchase' : 'Not Enough Points'}
                        </button>
                    </div>
                `).join('');
            } catch (error) {
                showAlert('Failed to load store', 'error');
            }
        }

        function getCategoryIcon(category) {
            const icons = {
                'theme': '🎨',
                'meditation': '🧘',
                'resource': '📚',
                'badge': '⭐',
                'premium': '👑'
            };
            return icons[category] || '🎁';
        }

        async function purchaseItem(itemId, cost, name) {
            if (!confirm(`Purchase "${name}" for ${cost} points?`)) return;

            try {
                const response = await fetch('/api/store/purchase', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ item_id: itemId })
                });

                const result = await response.json();

                if (result.success) {
                    showAlert(result.message, 'success');
                    loadWellnessStore();
                    loadPointsDisplay();
                } else {
                    showAlert(result.message, 'error');
                }
            } catch (error) {
                showAlert('Purchase failed', 'error');
            }
        }

        async function sendBuddyRequest() {
            const email = document.getElementById('buddy-email').value.trim();
            if (!email) {
                showAlert('Please enter an email', 'error');
                return;
            }

            try {
                const response = await fetch('/api/buddies/request', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ buddy_email: email })
                });

                const result = await response.json();

                if (response.ok) {
                    showAlert(result.message, 'success');
                    document.getElementById('buddy-email').value = '';
                } else {
                    showAlert(result.detail || 'Failed to send request', 'error');
                }
            } catch (error) {
                showAlert('Failed to send buddy request', 'error');
            }
        }

        async function loadBuddies() {
            if (!authToken) return;

            try {
                const response = await fetch('/api/buddies', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const data = await response.json();

                const pendingDiv = document.getElementById('pending-buddy-requests');
                if (data.pending_requests.length > 0) {
                    pendingDiv.innerHTML = `
                        <h3 style="margin-bottom: 15px;">📬 Pending Requests</h3>
                        ${data.pending_requests.map(req => `
                            <div style="background: rgba(255,255,255,0.08); padding: 20px; border-radius: 15px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>${req.name}</strong> wants to be your wellness buddy
                                    <div style="opacity: 0.7; font-size: 0.9rem; margin-top: 5px;">
                                        ${new Date(req.created_at).toLocaleDateString()}
                                    </div>
                                </div>
                                <button class="btn" style="width: auto; padding: 10px 20px;" onclick="acceptBuddyRequest(${req.id})">
                                    Accept
                                </button>
                            </div>
                        `).join('')}
                    `;
                } else {
                    pendingDiv.innerHTML = '';
                }

                const buddiesDiv = document.getElementById('my-buddies-list');
                if (data.buddies.length > 0) {
                    buddiesDiv.innerHTML = `
                        <h3 style="margin-bottom: 15px;">👥 Your Wellness Buddies</h3>
                        ${data.buddies.map(buddy => `
                            <div class="stat-card" style="text-align: left; cursor: pointer;" onclick="viewBuddyStats(${buddy.buddy_id})">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <div style="width: 50px; height: 50px; border-radius: 50%; background: linear-gradient(135deg, var(--accent), var(--success)); display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                                        👤
                                    </div>
                                    <div style="flex: 1;">
                                        <h3 style="margin-bottom: 5px;">${buddy.name}</h3>
                                        <div style="opacity: 0.7; font-size: 0.9rem;">
                                            Connected since ${new Date(buddy.accepted_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                    <i class="fas fa-chevron-right"></i>
                                </div>
                            </div>
                        `).join('')}
                    `;
                } else {
                    buddiesDiv.innerHTML = '<p style="text-align: center; opacity: 0.6;">No buddies yet. Send a request to connect!</p>';
                }
            } catch (error) {
                showAlert('Failed to load buddies', 'error');
            }
        }

        async function acceptBuddyRequest(requestId) {
            try {
                const response = await fetch(`/api/buddies/${requestId}/accept`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                if (response.ok) {
                    showAlert('Buddy request accepted! 🎉', 'success');
                    loadBuddies();
                }
            } catch (error) {
                showAlert('Failed to accept request', 'error');
            }
        }

        async function viewBuddyStats(buddyId) {
            try {
                const response = await fetch(`/api/buddies/${buddyId}/stats`, {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const data = await response.json();

                if (data.success) {
                    const modal = document.createElement('div');
                    modal.style.cssText = `
                        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                        background: rgba(0,0,0,0.8); display: flex; align-items: center;
                        justify-content: center; z-index: 1000;
                    `;
                    modal.innerHTML = `
                        <div style="background: var(--bg-light); padding: 40px; border-radius: 25px; max-width: 500px; position: relative;">
                            <button onclick="this.parentElement.parentElement.remove()" style="position: absolute; top: 15px; right: 15px; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer;">×</button>
                            <h2 style="margin-bottom: 30px;">${data.buddy_anonymous_name}'s Progress</h2>

                            <div style="display: grid; gap: 20px;">
                                <div class="stat-card">
                                    <div style="font-size: 3rem; margin-bottom: 10px;">🔥</div>
                                    <div class="stat-number">${data.current_streak}</div>
                                    <div style="margin-top: 10px;">Day Streak</div>
                                </div>

                                <div class="stat-card">
                                    <div style="font-size: 3rem; margin-bottom: 10px;">📊</div>
                                    <div class="stat-number">${data.logs_this_week}</div>
                                    <div style="margin-top: 10px;">Logs This Week</div>
                                </div>

                                <div class="stat-card">
                                    <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
                                    <div class="stat-number">${data.goals_completed_today}</div>
                                    <div style="margin-top: 10px;">Goals Today</div>
                                </div>
                            </div>

                            <div style="margin-top: 30px; padding: 20px; background: rgba(0,201,167,0.1); border-radius: 15px; text-align: center;">
                                Keep up the great work together! 💪
                            </div>
                        </div>
                    `;
                    document.body.appendChild(modal);
                }
            } catch (error) {
                showAlert('Failed to load buddy stats', 'error');
            }
        }

        async function createMicroHabit() {
            const goalType = document.getElementById('micro-habit-goal').value;
            const anchorHabit = document.getElementById('micro-habit-anchor').value;

            try {
                const response = await fetch('/api/habits/micro-stack', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({
                        goal_type: goalType,
                        anchor_habit: anchorHabit
                    })
                });

                const data = await response.json();

                if (data.anchor && data.micro_habit) {
                    const suggestionDiv = document.getElementById('micro-habit-suggestion');
                    suggestionDiv.innerHTML = `
                        <div style="background: linear-gradient(135deg, rgba(0,201,167,0.2) 0%, rgba(66,133,244,0.2) 100%); padding: 30px; border-radius: 20px; border: 2px solid var(--accent);">
                            <h3 style="color: var(--accent); margin-bottom: 20px;">✨ Your Micro Habit Stack</h3>

                            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                                <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 10px;">
                                    "${data.micro_habit}"
                                </div>
                            </div>

                            <div style="opacity: 0.9; margin-bottom: 20px; line-height: 1.6;">
                                <strong>Why this works:</strong> ${data.why_it_works}
                            </div>

                            <button class="btn" onclick="saveAndTrackMicroHabit()">
                                Save & Start Tracking
                            </button>
                        </div>
                    `;
                    suggestionDiv.style.display = 'block';
                }
            } catch (error) {
                showAlert('Failed to create micro habit', 'error');
            }
        }

        async function saveAndTrackMicroHabit() {
            showAlert('Micro habit saved! Track it daily to build your streak. 🌱', 'success');
            loadMicroHabits();
        }

        async function loadMicroHabits() {
            if (!authToken) return;

            try {
                const response = await fetch('/api/habits/micro-stack', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                const data = await response.json();

                const container = document.getElementById('my-micro-habits');

                if (data.micro_habits.length > 0) {
                    container.innerHTML = `
                        <h3 style="margin-bottom: 20px;">🌱 Your Active Micro Habits</h3>
                        ${data.micro_habits.map(habit => `
                            <div class="goal-card">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div style="flex: 1;">
                                        <h4 style="margin-bottom: 10px;">${habit.new_habit}</h4>
                                        <div style="opacity: 0.7; font-size: 0.9rem; margin-bottom: 10px;">
                                            Anchored to: ${habit.anchor_habit}
                                        </div>
                                        <div style="background: var(--accent); display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 700;">
                                            ${habit.completed_days} days completed
                                        </div>
                                    </div>
                                    <button class="btn" style="width: auto; padding: 10px 20px;" onclick="completeMicroHabitToday(${habit.id})">
                                        <i class="fas fa-check"></i> Done Today
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    `;
                } else {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.6;">No micro habits yet. Create your first one above!</p>';
                }
            } catch (error) {
                console.error('Failed to load micro habits:', error);
            }
        }

        async function completeMicroHabitToday(habitId) {
            try {
                const response = await fetch(`/api/habits/micro-stack/${habitId}/complete`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                const result = await response.json();

                if (result.success) {
                    showAlert(result.message, 'success');
                    loadMicroHabits();
                    loadPointsDisplay();
                }
            } catch (error) {
                showAlert('Failed to mark habit complete', 'error');
            }
        }

        async function loadGroupChallenges() {
            try {
                const response = await fetch('/api/challenges/group');
                const data = await response.json();

                const container = document.getElementById('group-challenges-list');

                if (data.group_challenges.length > 0) {
                    container.innerHTML = data.group_challenges.map(challenge => `
                        <div class="stat-card" style="text-align: left;">
                            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                                <div>
                                    <h3 style="margin-bottom: 5px;">${challenge.name}</h3>
                                    <div style="opacity: 0.7; font-size: 0.9rem;">by ${challenge.creator_name}</div>
                                </div>
                                <div style="background: var(--accent); padding: 5px 15px; border-radius: 20px; font-weight: 700;">
                                    ${challenge.member_count} members
                                </div>
                            </div>

                            <p style="margin-bottom: 15px;">${challenge.description}</p>

                            <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                                <div style="background: rgba(255,255,255,0.1); padding: 10px 15px; border-radius: 10px; flex: 1;">
                                    <div style="opacity: 0.7; font-size: 0.8rem;">Goal</div>
                                    <div style="font-weight: 700;">${challenge.goal_type}: ${challenge.target_value}</div>
                                </div>
                                <div style="background: rgba(255,255,255,0.1); padding: 10px 15px; border-radius: 10px; flex: 1;">
                                    <div style="opacity: 0.7; font-size: 0.8rem;">Ends</div>
                                    <div style="font-weight: 700;">${new Date(challenge.end_date).toLocaleDateString()}</div>
                                </div>
                            </div>

                            <div style="display: flex; gap: 10px;">
                                <button class="btn" style="flex: 1;" onclick="joinGroupChallenge(${challenge.id})">
                                    Join Challenge
                                </button>
                                <button class="btn" style="flex: 1; background: var(--glass-bg);" onclick="viewGroupLeaderboard(${challenge.id})">
                                    Leaderboard
                                </button>
                            </div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.6;">No active group challenges. Be the first to create one!</p>';
                }
            } catch (error) {
                showAlert('Failed to load group challenges', 'error');
            }
        }

        function showCreateGroupChallenge() {
            const form = document.getElementById('create-group-challenge-form');
            form.style.display = 'block';
            form.innerHTML = `
                <div class="form-container" style="max-width: 700px; margin-bottom: 30px;">
                    <h3 style="margin-bottom: 20px;">Create Group Challenge</h3>
                    <div class="form-group">
                        <label>Challenge Name</label>
                        <input type="text" id="gc-name" placeholder="e.g., 30-Day Water Challenge">
                    </div>
                    <div class="form-group">
                        <label>Description</label>
                        <input type="text" id="gc-description" placeholder="What's the challenge about?">
                    </div>
                    <div class="form-group">
                        <label>Goal Type</label>
                        <select id="gc-goal-type">
                            <option value="steps">Steps</option>
                            <option value="water">Water Intake</option>
                            <option value="sleep">Sleep Hours</option>
                            <option value="meditation">Meditation Minutes</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Target Value (daily)</label>
                        <input type="number" id="gc-target" step="0.1" placeholder="e.g., 10000 for steps">
                    </div>
                    <div class="form-group">
                        <label>Duration (days)</label>
                        <input type="number" id="gc-duration" value="30">
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn" onclick="submitGroupChallenge()">Create Challenge</button>
                        <button class="btn" style="background: var(--glass-bg);" onclick="document.getElementById('create-group-challenge-form').style.display='none'">Cancel</button>
                    </div>
                </div>
            `;
        }

        async function submitGroupChallenge() {
            const data = {
                name: document.getElementById('gc-name').value,
                description: document.getElementById('gc-description').value,
                goal_type: document.getElementById('gc-goal-type').value,
                target_value: parseFloat(document.getElementById('gc-target').value),
                duration_days: parseInt(document.getElementById('gc-duration').value)
            };

            if (!data.name || !data.target_value) {
                showAlert('Please fill all fields', 'error');
                return;
            }

            try {
                const response = await fetch('/api/challenges/group/create', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    showAlert(result.message, 'success');
                    document.getElementById('create-group-challenge-form').style.display = 'none';
                    loadGroupChallenges();
                }
            } catch (error) {
                showAlert('Failed to create challenge', 'error');
            }
        }

        async function joinGroupChallenge(groupId) {
            try {
                const response = await fetch(`/api/challenges/group/${groupId}/join`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                const result = await response.json();

                if (result.success) {
                    showAlert(result.message, 'success');
                    loadGroupChallenges();
                } else {
                    showAlert(result.message, 'error');
                }
            } catch (error) {
                showAlert('Failed to join challenge', 'error');
            }
        }

        async function viewGroupLeaderboard(groupId) {
            try {
                const response = await fetch(`/api/challenges/group/${groupId}/leaderboard`);
                const data = await response.json();

                const modal = document.createElement('div');
                modal.style.cssText = `
                    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                    background: rgba(0,0,0,0.8); display: flex; align-items: center;
                    justify-content: center; z-index: 1000; overflow-y: auto;
                `;
                modal.innerHTML = `
                    <div style="background: var(--bg-light); padding: 40px; border-radius: 25px; max-width: 600px; max-height: 80vh; overflow-y: auto; position: relative;">
                        <button onclick="this.parentElement.parentElement.remove()" style="position: absolute; top: 15px; right: 15px; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer;">×</button>
                        <h2 style="margin-bottom: 30px;">🏆 Challenge Leaderboard</h2>

                        ${data.leaderboard.map((entry, index) => `
                            <div style="background: rgba(255,255,255,0.08); padding: 20px; border-radius: 15px; margin-bottom: 15px; display: flex; align-items: center; gap: 20px;">
                                <div style="font-size: 2rem; font-weight: 700; ${index < 3 ? 'color: var(--accent);' : ''}">
                                    ${index + 1}${index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : ''}
                                </div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 700; margin-bottom: 5px;">${entry.name}</div>
                                    <div style="opacity: 0.7; font-size: 0.9rem;">
                                        Avg: ${entry.avg_value ? entry.avg_value.toFixed(1) : 0} | Logged: ${entry.days_logged} days
                                    </div>
                                </div>
                            </div>
                        `).join('')}

                        ${data.leaderboard.length === 0 ? '<p style="text-align: center; opacity: 0.6;">No data yet. Start logging to appear on the leaderboard!</p>' : ''}
                    </div>
                `;
                document.body.appendChild(modal);
            } catch (error) {
                showAlert('Failed to load leaderboard', 'error');
            }
        }

        async function generateSoundscape(emotion) {
            try {
                const response = await fetch('/api/wellness-vibe/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ emotion })
                });

                const data = await response.json();
                const soundscape = data.soundscape;

                const player = document.getElementById('soundscape-player');
                player.innerHTML = `
                    <div style="background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%); padding: 40px; border-radius: 25px; border: 2px solid rgba(102,126,234,0.3);">
                        <h2 style="color: var(--accent); margin-bottom: 20px;">🎵 ${soundscape.type}</h2>

                        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; margin-bottom: 20px;">
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
                                <div>
                                    <div style="opacity: 0.7; font-size: 0.9rem;">Frequency</div>
                                    <div style="font-weight: 700; font-size: 1.2rem;">${soundscape.frequency}</div>
                                </div>
                                <div>
                                    <div style="opacity: 0.7; font-size: 0.9rem;">Duration</div>
                                    <div style="font-weight: 700; font-size: 1.2rem;">${soundscape.duration}</div>
                                </div>
                            </div>

                            <div style="margin-bottom: 20px;">
                                <div style="opacity: 0.7; font-size: 0.9rem; margin-bottom: 10px;">Sound Elements:</div>
                                ${soundscape.elements.map(el => `
                                    <div style="background: rgba(0,201,167,0.2); padding: 8px 15px; border-radius: 20px; display: inline-block; margin: 5px;">
                                        ${el}
                                    </div>
                                `).join('')}
                            </div>
                        </div>

                        <div style="background: rgba(0,201,167,0.1); padding: 20px; border-radius: 15px; margin-bottom: 25px;">
                            <h4 style="margin-bottom: 10px;">💡 Why This Works</h4>
                            <p style="line-height: 1.6;">${soundscape.reasoning}</p>
                        </div>

                        <div style="text-align: center; padding: 40px; background: rgba(0,0,0,0.3); border-radius: 15px; margin-bottom: 20px;">
                            <div style="font-size: 4rem; margin-bottom: 20px; animation: pulse 2s ease-in-out infinite;">
                                🎧
                            </div>
                            <p style="opacity: 0.8;">In a real implementation, your personalized ${soundscape.duration} soundscape would play here.</p>
                            <p style="font-size: 0.9rem; opacity: 0.6; margin-top: 10px;">
                                (Integration with audio generation libraries like Tone.js or Web Audio API required)
                            </p>
                        </div>

                        <button class="btn" onclick="document.getElementById('soundscape-player').style.display='none'">
                            Close
                        </button>
                    </div>
                `;
                player.style.display = 'block';
                player.scrollIntoView({ behavior: 'smooth' });

            } catch (error) {
                showAlert('Failed to generate soundscape', 'error');
            }
        }

        async function exportDataJSON() {
            try {
                const response = await fetch('/api/data/export', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `wellness_data_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                showAlert('Data exported successfully! 📥', 'success');
            } catch (error) {
                showAlert('Failed to export data', 'error');
            }
        }

        async function exportDataCSV() {
            try {
                const response = await fetch('/api/data/export/csv', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `health_logs_${new Date().toISOString().split('T')[0]}.csv`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                showAlert('Health logs exported successfully! 📥', 'success');
            } catch (error) {
                showAlert('Failed to export CSV', 'error');
            }
        }

        function logout() {
            currentUser = null;
            authToken = null;
            showAlert('Logged out successfully', 'success');
            showScreen('main-menu');
            updateMainMenu();
            document.getElementById('points-display').style.display = 'none';
            document.getElementById('nudges-banner').style.display = 'none';
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadDailyQuote();
            updateMainMenu();

            if (authToken) {
                loadPointsDisplay();
                loadNudgesBanner();
            }
        });
    </script>
</body>
</html>
'''

# ============================================================================
# SERVER STARTUP
# ============================================================================

def wait_for_port(host, port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False

def start_uvicorn():
    from uvicorn import Config, Server
    config = Config(app=app, host="0.0.0.0", port=8001, log_level="info")
    server = Server(config=config)
    server.run()

def run_server():
    print("\n" + "="*70)
    print("🚀 ENHANCED GLOBAL WELLNESS HEALTH CHATBOT v6.0")
    print("="*70)
    print("\n📋 MERGED FEATURES:")
    print("  ✓ Multilingual AI Chatbot (Auto-detect language)")
    print("  ✓ Health Logger with analytics")
    print("  ✓ Wellness Goals & Streak Tracking")
    print("  ✓ AI Diet Planning (Vegetarian/Non-veg/Vegan)")
    print("  ✓ Personalized Fitness Plans")
    print("  ✓ Community Forum")
    print("  ✓ Wellness Journal with Emotion Analysis")
    print("  ✓ Meditation & Mindfulness Hub")
    print("  ✓ Daily Challenges & Achievements")
    print("  ✓ Personalized Health Insights")
    print("  ✓ Crisis Detection & Support")
    print("  ✓ Real-time Analytics Dashboard")
    print("\n" + "="*70)
    print("⏳ Starting server...")
    print("="*70 + "\n")

    # Initialize nudge scheduler AFTER server starts
    def delayed_init():
        time.sleep(3)  # Wait for server to be ready
        global nudge_scheduler
        nudge_scheduler = ProactiveNudgeScheduler(db_manager, health_chatbot)
        nudge_scheduler.start()

    threading.Thread(target=delayed_init, daemon=True).start()

    server_thread = threading.Thread(target=start_uvicorn, daemon=True)
    server_thread.start()

    print("⏳ Waiting for FastAPI to start...")
    if wait_for_port("127.0.0.1", 8001):
        print("✅ FastAPI is running!")
    else:
        print("❌ FastAPI failed to start on port 8001")
        return

    try:
        ngrok.set_auth_token("34s01psdiRyxZf5F1EgE8GaqCcg_7PQUKk8whGh2RLmyGTzyE")
        public_url = ngrok.connect(8001)
        print(f"\n🌐 Public URL: {public_url}")
    except Exception as e:
        print("⚠️ Ngrok connection failed.")
        print(f"Error: {e}")

    print(f"📍 Local URL: http://localhost:8001")
    print("\n" + "="*70)
    print("✅ SERVER IS LIVE AND READY!")
    print("="*70 + "\n")

    server_thread.join()

if __name__ == "__main__":
    import os
    
    # Check if running on Hugging Face
    if os.getenv("SPACE_ID"):
        # Hugging Face Spaces environment
        print("\n🚀 Running on Hugging Face Spaces...")
        from uvicorn import Config, Server
        config = Config(app=app, host="0.0.0.0", port=7860, log_level="info")
        server = Server(config=config)
        server.run()
    else:
        # Local development with ngrok
        run_server()
