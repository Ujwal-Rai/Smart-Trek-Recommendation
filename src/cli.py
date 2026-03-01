# ============================================================
# cli.py
# Project: AI-Based Smart Trek Recommendation System for Nepal
# Purpose: Command Line Interface — collects user preferences
#          and outputs the Top 3 recommended trekking routes.
# Run with: python cli.py
# ============================================================

import sys
import os

# Add src directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_data, build_user_vector, calculate_similarity, rank_treks, display_recommendations

# ============================================================
# PATH CONFIGURATION
# Update this to match your actual file location
# ============================================================

DATA_PATH = r"E:\AI Project for third sem\Trek Data.csv"


# ============================================================
# INPUT HELPER FUNCTIONS
# Each function handles one user input with validation.
# If the user enters something invalid, they are prompted
# again until a valid value is entered.
# ============================================================

def get_duration():
    """
    Asks the user how many days they have available.
    Validates that input is a positive integer.
    """
    while True:
        try:
            days = int(input("  Enter number of days available (e.g. 7, 14, 21): "))
            if days <= 0:
                print("  ⚠️  Please enter a positive number of days.\n")
            elif days > 30:
                print("  ⚠️  Maximum supported duration is 30 days.\n")
            else:
                return days
        except ValueError:
            print("  ⚠️  Invalid input. Please enter a whole number (e.g. 14).\n")


def get_budget():
    """
    Asks the user for their total budget in USD.
    Validates that input is a positive number.
    """
    while True:
        try:
            budget = float(input("  Enter your total budget in USD (e.g. 1000, 1500): $"))
            if budget <= 0:
                print("  ⚠️  Budget must be greater than 0.\n")
            else:
                return budget
        except ValueError:
            print("  ⚠️  Invalid input. Please enter a number (e.g. 1500).\n")


def get_difficulty():
    """
    Asks the user to choose a difficulty level from 3 options.
    Validates against allowed choices.
    """
    options = {'1': 'Easy', '2': 'Moderate', '3': 'Hard'}

    while True:
        print("  Difficulty levels:")
        print("    1 → Easy      (suitable for beginners, light hiking)")
        print("    2 → Moderate  (some trekking experience recommended)")
        print("    3 → Hard      (experienced trekkers, high altitude)")
        choice = input("  Enter your choice (1, 2 or 3): ").strip()

        if choice in options:
            return options[choice]
        else:
            print("  ⚠️  Invalid choice. Please enter 1, 2 or 3.\n")


def get_accommodation():
    """
    Asks the user to choose their preferred accommodation type.
    """
    options = {'1': 'Guesthouse', '2': 'Teahouse', '3': 'Lodge'}

    while True:
        print("  Accommodation types:")
        print("    1 → Guesthouse  (hotels and guesthouses in towns)")
        print("    2 → Teahouse    (basic mountain tea houses on trail)")
        print("    3 → Lodge       (lodges and luxury lodges)")
        choice = input("  Enter your choice (1, 2 or 3): ").strip()

        if choice in options:
            return options[choice]
        else:
            print("  ⚠️  Invalid choice. Please enter 1, 2 or 3.\n")


def get_season():
    """
    Asks the user to choose their preferred trekking season.
    """
    options = {
        '1': 'Spring & Autumn',
        '2': 'Spring',
        '3': 'Autumn'
    }

    while True:
        print("  Preferred trekking season:")
        print("    1 → Spring & Autumn  (March-May & Sept-Dec, most popular)")
        print("    2 → Spring only      (March - May, flowers in bloom)")
        print("    3 → Autumn only      (Sept - December, clear skies)")
        choice = input("  Enter your choice (1, 2 or 3): ").strip()

        if choice in options:
            return options[choice]
        else:
            print("  ⚠️  Invalid choice. Please enter 1, 2 or 3.\n")


# ============================================================
# WELCOME BANNER
# ============================================================

def print_banner():
    """Prints the welcome banner when the CLI starts."""
    print("\n")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     🏔️   NEPAL SMART TREK RECOMMENDATION SYSTEM      ║")
    print("  ║         AI-Powered Trekking Route Advisor            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print("  Welcome! This system will recommend the best trekking")
    print("  routes in Nepal based on your personal preferences.")
    print()
    print("  Please answer the following questions:")
    print("  " + "─" * 54)


# ============================================================
# CONFIRM USER INPUT
# Shows a summary of what the user entered and asks them
# to confirm before running the recommendation.
# ============================================================

def confirm_input(user_input):
    """
    Displays a summary of user preferences and asks for confirmation.
    Returns True if confirmed, False if user wants to re-enter.
    """
    print("\n  " + "─" * 54)
    print("  📋  YOUR PREFERENCES SUMMARY")
    print("  " + "─" * 54)
    print(f"  • Days Available  : {user_input['duration_days']} days")
    print(f"  • Budget          : ${user_input['cost_usd']:,.0f} USD")
    print(f"  • Difficulty      : {user_input['difficulty_level']}")
    print(f"  • Accommodation   : {user_input['accommodation']}")
    print(f"  • Season          : {user_input['best_season']}")
    print("  " + "─" * 54)

    while True:
        confirm = input("\n  Is this correct? (yes / no): ").strip().lower()
        if confirm in ['yes', 'y']:
            return True
        elif confirm in ['no', 'n']:
            return False
        else:
            print("  ⚠️  Please type 'yes' or 'no'.")


# ============================================================
# ASK TO RUN AGAIN
# After showing results, ask if the user wants to try
# a different profile.
# ============================================================

def ask_run_again():
    """Asks the user if they want to search with a new profile."""
    while True:
        again = input("\n  Would you like to search with a different profile? (yes / no): ").strip().lower()
        if again in ['yes', 'y']:
            return True
        elif again in ['no', 'n']:
            return False
        else:
            print("  ⚠️  Please type 'yes' or 'no'.")


# ============================================================
# MAIN CLI FUNCTION
# Orchestrates the full user interaction loop.
# ============================================================

def main():
    """
    Main CLI loop:
    1. Show welcome banner
    2. Load data (once, before the loop)
    3. Collect user inputs with validation
    4. Confirm inputs
    5. Run recommendation engine
    6. Display top 3 results
    7. Ask to run again or exit
    """

    # --- Show banner ---
    print_banner()

    # --- Load and preprocess data ONCE before the loop ---
    # This avoids reloading the dataset on every run
    print("  ⏳ Loading trek database, please wait...")
    try:
        df, df_scaled, feature_cols, scaler = load_data(DATA_PATH)
        print(f"  ✅ Database loaded: {len(df)} trekking routes ready.\n")
    except FileNotFoundError:
        print(f"\n  ❌ ERROR: Dataset not found at:\n     {DATA_PATH}")
        print("  Please update the DATA_PATH variable in cli.py")
        print("  to point to your Trek Data.csv file location.\n")
        sys.exit(1)

    # --- Main interaction loop ---
    while True:

        # Collect inputs — loop until user confirms
        while True:
            print("\n  STEP 1 — How many days do you have?")
            duration = get_duration()

            print("\n  STEP 2 — What is your budget?")
            budget = get_budget()

            print("\n  STEP 3 — What is your fitness/difficulty level?")
            difficulty = get_difficulty()

            print("\n  STEP 4 — What type of accommodation do you prefer?")
            accommodation = get_accommodation()

            print("\n  STEP 5 — What is your preferred trekking season?")
            season = get_season()

            # Build the user input dictionary
            user_input = {
                'duration_days'   : duration,
                'cost_usd'        : budget,
                'difficulty_level': difficulty,
                'accommodation'   : accommodation,
                'best_season'     : season
            }

            # Confirm with user
            if confirm_input(user_input):
                break  # confirmed — move to recommendation
            else:
                print("\n  🔄 Let's start over. Please re-enter your preferences.\n")

        # --- Run recommendation engine ---
        print("\n  ⏳ Finding your best treks...")

        user_vector       = build_user_vector(user_input, feature_cols, scaler)
        similarity_scores = calculate_similarity(user_vector, df_scaled, feature_cols)
        results           = rank_treks(df, similarity_scores, top_n=3)

        # --- Display results ---
        display_recommendations(results, user_input)

        # --- Ask to run again or exit ---
        if not ask_run_again():
            print("\n  " + "═" * 54)
            print("  🙏 Thank you for using the Nepal Trek Recommender!")
            print("  Safe travels and happy trekking! 🏔️")
            print("  " + "═" * 54 + "\n")
            break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()