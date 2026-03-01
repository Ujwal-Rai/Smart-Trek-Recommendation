import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import load_data, build_user_vector, calculate_similarity, rank_treks, display_recommendations

DATA_PATH = r"E:\AI Project for third sem\Trek Data.csv"

def get_duration():

    while True:
        try:
            days = int(input("  Enter number of days available (e.g. 7, 14, 21): "))
            if days <= 0:
                print("  Please enter a positive number of days.\n")
            elif days > 30:
                print("  Maximum supported duration is 30 days.\n")
            else:
                return days
        except ValueError:
            print("   Invalid input. Please enter a whole number (e.g. 14).\n")


def get_budget():

    while True:
        try:
            budget = float(input("  Enter your total budget in USD (e.g. 1000, 1500): $"))
            if budget <= 0:
                print("   Budget must be greater than 0.\n")
            else:
                return budget
        except ValueError:
            print("    Invalid input. Please enter a number (e.g. 1500).\n")


def get_difficulty():

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
            print("    Invalid choice. Please enter 1, 2 or 3.\n")


def get_accommodation():

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
            print("    Invalid choice. Please enter 1, 2 or 3.\n")


def get_season():
 
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
            print("    Invalid choice. Please enter 1, 2 or 3.\n")


def print_banner():

    print("\n")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║        NEPAL SMART TREK RECOMMENDATION SYSTEM      ║")
    print("  ║         AI-Powered Trekking Route Advisor            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print("  Welcome! This system will recommend the best trekking")
    print("  routes in Nepal based on your personal preferences.")
    print()
    print("  Please answer the following questions:")
    print("  " + "─" * 54)



def confirm_input(user_input):
    """
    Displays a summary of user preferences and asks for confirmation.
    Returns True if confirmed, False if user wants to re-enter.
    """
    print("\n  " + "─" * 54)
    print("   YOUR PREFERENCES SUMMARY")
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
            print("   Please type 'yes' or 'no'.")



def ask_run_again():
    """Asks the user if they want to search with a new profile."""
    while True:
        again = input("\n  Would you like to search with a different profile? (yes / no): ").strip().lower()
        if again in ['yes', 'y']:
            return True
        elif again in ['no', 'n']:
            return False
        else:
            print("   Please type 'yes' or 'no'.")



def main():

 
    print_banner()

 
    print("   Loading trek database, please wait...")
    try:
        df, df_scaled, feature_cols, scaler = load_data(DATA_PATH)
        print(f"   Database loaded: {len(df)} trekking routes ready.\n")
    except FileNotFoundError:
        print(f"\n   ERROR: Dataset not found at:\n     {DATA_PATH}")
        print("  Please update the DATA_PATH variable in cli.py")
        print("  to point to your Trek Data.csv file location.\n")
        sys.exit(1)

  
    while True:

       
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

            
            user_input = {
                'duration_days'   : duration,
                'cost_usd'        : budget,
                'difficulty_level': difficulty,
                'accommodation'   : accommodation,
                'best_season'     : season
            }

            
            if confirm_input(user_input):
                break 
            else:
                print("\n   Let's start over. Please re-enter your preferences.\n")

        
        print("\n   Finding your best treks...")

        user_vector       = build_user_vector(user_input, feature_cols, scaler)
        similarity_scores = calculate_similarity(user_vector, df_scaled, feature_cols)
        results           = rank_treks(df, similarity_scores, top_n=3)

        
        display_recommendations(results, user_input)

        
        if not ask_run_again():
            print("\n  " + "═" * 54)
            print("  Thank you for using the Nepal Trek Recommender!")
            print("  Safe travels and happy trekking! ")
            print("  " + "═" * 54 + "\n")
            break



if __name__ == "__main__":
    main()