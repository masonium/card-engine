extern crate card_engine;
extern crate rand;

use card_engine::{Round, GameState, Action, ActionError};
use card_engine::cards::BasicCard;
use rand::{thread_rng};
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

fn print_round(round: &Round) {
    println!("{}", &format_state(round.get_state()));
    println!("{}\n", round.get_phase());
}

fn print_action(action: &Action) {
    println!("Player {} plays {}.", action.player + 1, action.card);
}


/// Randomly choose actions at each play
fn play_random_game(verbose: bool) -> Result<[usize; 2], ActionError> {
    let mut round = Round::new();
    let mut rng = thread_rng();

    if verbose {
        print_round(&round);
    }

    let mut actions = round.possible_actions();

    while actions.len() > 0 {

        let card_range = Range::new(0, actions.len());
        let ri = card_range.ind_sample(&mut rng) as usize;

        let action = actions[ri].clone();
        if verbose {
            print_action(&action);
        }
        round.play_action(action)?;

        if verbose {
            println!("**************");
            print_round(&round);
        }

        actions = round.possible_actions();
    }

    Ok(round.get_state().score.clone())
}

fn main() {
    let mut games_won = [0, 0];
    for _ in 0..100 {
        let score = play_random_game(false).expect("bad game");
        let winner: usize = if score[0] > score[1]  { 0 } else { 1 };
        games_won[winner] += 1;
    }

    println!("Player 1 Record: {}-{}", games_won[0], games_won[1]);
}
