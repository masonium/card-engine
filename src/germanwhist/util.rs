use super::engine::{Action, Round};
use super::state::GameState;
use crate::cards::BasicCard;

pub fn format_hand(hand: &[BasicCard], gs: &GameState) -> String {
    let mut cards: Vec<_> = hand.iter().collect();
    cards.sort_by(|a, b| gs.display_order(a, b));

    let sc: Vec<_> = cards.iter().map(|x| format!("{}", x)).collect();
    sc.join(" ")
}

pub fn format_state(gs: &GameState) -> String {
    format!(
        "Player 1: Score {}, Hand: {}\nPlayer 2: Score {}, Hand: {}\nTrump: {}\n",
        gs.score[0],
        &format_hand(&gs.hands[0], gs),
        gs.score[1],
        &format_hand(&gs.hands[1], gs),
        gs.trump
    )
}

pub fn format_round(round: &Round) -> String {
    format!(
        "{}{}\n",
        &format_state(round.get_state()),
        round.get_phase().format(round.get_state())
    )
}

pub fn format_action(action: &Action) -> String {
    format!("Player {} plays {}.", action.player + 1, action.card)
}
