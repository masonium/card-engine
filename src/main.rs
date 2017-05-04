extern crate card_engine;
extern crate rand;

use card_engine::{Round, GameState, Action};
use card_engine::cards::BasicCard;
use rand::{Rand, Rng, thread_rng};
use rand::distributions::{IndependentSample, Range};

fn format_hand(hand: &[BasicCard], gs: &GameState) -> String {
    let mut cards: Vec<_> = hand.iter().collect();
    cards.sort_by(|a, b| gs.display_order(a, b));

    let sc: Vec<_> = cards.iter().map(|x| format!("{}", x)).collect();
    sc.join(" ")
}

fn format_state(gs: &GameState) -> String {
    format!("Player 1: Score {}, Hand: {}\nPlayer 2: Score {}, Hand: {}\nTrump: {}",
            gs.score[0], &format_hand(&gs.hands[0], gs),
            gs.score[1], &format_hand(&gs.hands[1], gs),
            gs.trump)
}

fn format_round(round: &Round) {
    println!("{}", &format_state(round.get_state()));
    println!("{}\n", round.get_phase());
}


fn main() {
    let mut round = Round::new();
    let mut rng = thread_rng();

    format_round(&round);

    let card_range = Range::new(0, 13);
    let card = round.get_state().hands[0][card_range.ind_sample(&mut rng) as usize].clone();

    round.play_action(Action{ player: 0, card });

    format_round(&round);
}
