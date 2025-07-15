import random
from collections import defaultdict, deque
from typing import List, Dict, Optional

class PsychologicalRPSAI:
    def __init__(self):
        # Game state tracking
        self.opponent_history = []
        self.my_history = []
        self.outcomes = []  # 'win', 'lose', 'tie'
        self.round_count = 0
        
        # Strategy weights (importance-based)
        self.strategy_weights = {
            'win_stay_lose_shift': 0.35,
            'frequency_analysis': 0.25,
            'three_move_avoidance': 0.20,
            'clockwise_cycle': 0.10,
            'opening_strategy': 0.05,
            'recent_bias': 0.05
        }
        
        # Pattern tracking
        self.opponent_frequencies = {'R': 0, 'P': 0, 'S': 0}
        self.win_stay_patterns = {'after_win': defaultdict(int), 'after_loss': defaultdict(int)}
        self.recent_window = deque(maxlen=5)  # Track last 5 moves with higher weight
        
        # Moves mapping
        self.moves = {'R': 'Rock', 'P': 'Paper', 'S': 'Scissors'}
        self.counters = {'R': 'P', 'P': 'S', 'S': 'R'}
        self.beats = {'R': 'S', 'P': 'R', 'S': 'P'}
    
    def determine_outcome(self, my_move: str, opponent_move: str) -> str:
        """Determine the outcome from AI's perspective"""
        if my_move == opponent_move:
            return 'tie'
        elif self.beats[my_move] == opponent_move:
            return 'win'
        else:
            return 'lose'
    
    def update_patterns(self, opponent_move: str, outcome: str):
        """Update all pattern tracking data"""
        self.opponent_history.append(opponent_move)
        self.outcomes.append(outcome)
        self.recent_window.append(opponent_move)
        self.opponent_frequencies[opponent_move] += 1
        
        # Update win-stay/lose-shift patterns
        if len(self.outcomes) >= 2:
            last_outcome = self.outcomes[-2]
            if last_outcome in ['win', 'lose']:  # From opponent's perspective, reverse our outcome
                opp_last_outcome = 'lose' if last_outcome == 'win' else 'win'
                last_move = self.opponent_history[-2]
                current_move = opponent_move
                
                if opp_last_outcome == 'win':
                    if last_move == current_move:
                        self.win_stay_patterns['after_win']['stay'] += 1
                    else:
                        self.win_stay_patterns['after_win']['shift'] += 1
                else:  # opponent lost last round
                    if last_move == current_move:
                        self.win_stay_patterns['after_loss']['stay'] += 1
                    else:
                        self.win_stay_patterns['after_loss']['shift'] += 1
    
    def strategy_win_stay_lose_shift(self) -> Optional[str]:
        """Predict based on win-stay/lose-shift psychology"""
        if len(self.outcomes) < 1:
            return None
            
        last_outcome = self.outcomes[-1]
        if len(self.opponent_history) < 1:
            return None
            
        last_opponent_move = self.opponent_history[-1]
        
        # From opponent's perspective
        opponent_last_outcome = 'lose' if last_outcome == 'win' else ('win' if last_outcome == 'lose' else 'tie')
        
        if opponent_last_outcome == 'win':
            # Opponent likely to repeat (stay)
            return self.counters[last_opponent_move]
        elif opponent_last_outcome == 'lose':
            # Opponent likely to shift (clockwise cycle is common)
            shift_map = {'R': 'P', 'P': 'S', 'S': 'R'}
            predicted_move = shift_map[last_opponent_move]
            return self.counters[predicted_move]
        
        return None
    
    def strategy_frequency_analysis(self) -> str:
        """Counter the most frequent move + anti-Rock bias"""
        total_moves = sum(self.opponent_frequencies.values())
        if total_moves == 0:
            return 'P'  # Anti-Rock bias for opening
        
        # Find most frequent move
        most_frequent = max(self.opponent_frequencies.items(), key=lambda x: x[1])
        
        # Add anti-Rock bias (people favor Rock)
        rock_bias = 1.2  # Boost Rock frequency assumption
        adjusted_freq = dict(self.opponent_frequencies)
        adjusted_freq['R'] *= rock_bias
        
        most_likely = max(adjusted_freq.items(), key=lambda x: x[1])[0]
        return self.counters[most_likely]
    
    def strategy_three_move_avoidance(self) -> Optional[str]:
        """Predict based on humans avoiding 3+ consecutive moves"""
        if len(self.opponent_history) < 2:
            return None
        
        # Check if last two moves are the same
        if self.opponent_history[-1] == self.opponent_history[-2]:
            # Very likely to change, counter the most likely alternative
            last_move = self.opponent_history[-1]
            alternatives = [m for m in ['R', 'P', 'S'] if m != last_move]
            # Slight preference for clockwise shift
            clockwise_shift = {'R': 'P', 'P': 'S', 'S': 'R'}
            predicted = clockwise_shift.get(last_move, random.choice(alternatives))
            return self.counters[predicted]
        
        return None
    
    def strategy_clockwise_cycle(self) -> Optional[str]:
        """Detect clockwise cycling patterns"""
        if len(self.opponent_history) < 2:
            return None
        
        # Check for clockwise pattern
        clockwise_transitions = {'R': 'P', 'P': 'S', 'S': 'R'}
        
        # Look for clockwise transitions in recent history
        clockwise_count = 0
        for i in range(1, min(4, len(self.opponent_history))):
            prev_move = self.opponent_history[-i-1]
            curr_move = self.opponent_history[-i]
            if clockwise_transitions.get(prev_move) == curr_move:
                clockwise_count += 1
        
        if clockwise_count >= 1:  # Detected clockwise tendency
            last_move = self.opponent_history[-1]
            predicted = clockwise_transitions[last_move]
            return self.counters[predicted]
        
        return None
    
    def strategy_opening(self) -> str:
        """Opening move strategy - counter Rock bias"""
        if self.round_count == 0:
            return 'S'  # Scissors beats Rock (most common opening)
        elif self.round_count == 1:
            return 'P'  # Paper is statistically safest
        return None
    
    def strategy_recent_bias(self) -> Optional[str]:
        """Weight recent moves more heavily"""
        if len(self.recent_window) < 2:
            return None
        
        # Count recent moves with exponential weighting
        weighted_counts = {'R': 0, 'P': 0, 'S': 0}
        for i, move in enumerate(self.recent_window):
            weight = 2 ** i  # More recent = higher weight
            weighted_counts[move] += weight
        
        most_likely = max(weighted_counts.items(), key=lambda x: x[1])[0]
        return self.counters[most_likely]
    
    def get_ai_move(self) -> str:
        """Main decision function combining all strategies"""
        predictions = {}
        
        # Gather predictions from all strategies
        strategies = {
            'win_stay_lose_shift': self.strategy_win_stay_lose_shift(),
            'frequency_analysis': self.strategy_frequency_analysis(),
            'three_move_avoidance': self.strategy_three_move_avoidance(),
            'clockwise_cycle': self.strategy_clockwise_cycle(),
            'opening_strategy': self.strategy_opening(),
            'recent_bias': self.strategy_recent_bias()
        }
        
        # Weight the predictions
        move_scores = {'R': 0, 'P': 0, 'S': 0}
        
        for strategy_name, prediction in strategies.items():
            if prediction:
                weight = self.strategy_weights[strategy_name]
                move_scores[prediction] += weight
        
        # If no clear winner, use frequency analysis as fallback
        if max(move_scores.values()) == 0:
            return self.strategy_frequency_analysis()
        
        # Add small random element to avoid being too predictable
        for move in move_scores:
            move_scores[move] += random.uniform(0, 0.05)
        
        return max(move_scores.items(), key=lambda x: x[1])[0]
    
    def play_round(self, opponent_move: str) -> tuple[str, str]:
        """Play one round and return (ai_move, outcome)"""
        ai_move = self.get_ai_move()
        self.my_history.append(ai_move)
        
        outcome = self.determine_outcome(ai_move, opponent_move)
        self.update_patterns(opponent_move, outcome)
        
        self.round_count += 1
        return ai_move, outcome
    
    def get_stats(self) -> Dict:
        """Get current game statistics"""
        if not self.outcomes:
            return {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0, 'win_rate': 0}
        
        wins = self.outcomes.count('win')
        losses = self.outcomes.count('lose')
        ties = self.outcomes.count('tie')
        total = len(self.outcomes)
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        return {
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'total': total,
            'win_rate': win_rate
        }

def main():
    ai = PsychologicalRPSAI()
    print("🎮 Psychological Rock-Paper-Scissors AI Challenge!")
    print("🧠 I'm using psychology and pattern recognition to beat you!")
    print("📝 Enter: R (Rock), P (Paper), S (Scissors), or Q (Quit)")
    print("=" * 60)
    
    while True:
        # Get human move
        human_input = input(f"\nRound {ai.round_count + 1} - Your move: ").upper().strip()
        
        if human_input == 'Q':
            break
        
        if human_input not in ['R', 'P', 'S']:
            print("❌ Invalid input! Use R, P, S, or Q")
            continue
        
        # AI makes its move
        ai_move, outcome = ai.play_round(human_input)
        
        # Display results
        print(f"You played: {ai.moves[human_input]}")
        print(f"AI played: {ai.moves[ai_move]}")
        
        if outcome == 'win':
            print("🤖 AI wins this round!")
        elif outcome == 'lose':
            print("🎉 You win this round!")
        else:
            print("🤝 It's a tie!")
        
        # Show statistics
        stats = ai.get_stats()
        print(f"\n📊 AI Stats: {stats['wins']}W-{stats['losses']}L-{stats['ties']}T | Win Rate: {stats['win_rate']:.1f}%")
        
        # Show AI's analysis (every 5 rounds)
        if ai.round_count % 5 == 0 and ai.round_count > 0:
            print(f"\n🔍 AI Analysis after {ai.round_count} rounds:")
            freq_analysis = ai.opponent_frequencies
            total = sum(freq_analysis.values())
            if total > 0:
                for move, count in freq_analysis.items():
                    percentage = (count / total) * 100
                    print(f"   {ai.moves[move]}: {percentage:.1f}% ({count}/{total})")
    
    # Final statistics
    final_stats = ai.get_stats()
    print(f"\n🏁 Final Results:")
    print(f"   AI: {final_stats['wins']} wins")
    print(f"   You: {final_stats['losses']} wins")
    print(f"   Ties: {final_stats['ties']}")
    print(f"   AI Win Rate: {final_stats['win_rate']:.1f}%")
    
    if final_stats['win_rate'] > 50:
        print("🧠 Psychology wins! The AI successfully exploited your patterns.")
    elif final_stats['win_rate'] < 33:
        print("🎯 Impressive! You successfully avoided the AI's psychological traps.")
    else:
        print("⚖️ Close game! The psychological battle was evenly matched.")

if __name__ == "__main__":
    main()
