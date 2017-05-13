extern crate card_engine;
extern crate rand;
extern crate ndarray;

use card_engine::{GameEvent, Round, Action, ActionError};
use card_engine::cards::{self, BasicCard, Rank, Suit};
use card_engine::germanwhist::util::*;
use card_engine::germanwhist::PlayerView;
use card_engine::germanwhist::PlayerBelief;
// use card_engine::{NeuralNet, LayerDesc, OutputFunction};
use rand::{thread_rng};
use rand::distributions::{IndependentSample, Range};

trait Player {
    fn on_game_action(&mut self, _ev: &GameEvent) { }

    /// Return a card to play, based on the current view of the world.
    fn play_card(&self, view: &PlayerView) -> BasicCard;
}

pub struct RandomPlayer;

impl RandomPlayer {
    fn new() -> RandomPlayer {
        RandomPlayer { }
    }
}

impl Player for RandomPlayer {
    fn play_card(&self, view: &PlayerView) -> BasicCard {
        let mut rng = thread_rng();
        let cards = view.playable_cards();
        let card_range = Range::new(0, cards.len());
        let ri = card_range.ind_sample(&mut rng) as usize;
        cards[ri].clone()
    }
}

pub struct BasicPlayer {
    // try to win any non-trump with rank above this
    min_nontrump_rank_to_win: Option<u8>
}

impl BasicPlayer {
    fn new(mntr: Option<Rank>) -> BasicPlayer {
        BasicPlayer { min_nontrump_rank_to_win: mntr.map(|c| c.ord_ace_high()) }
    }

    fn try_to_win(&self, card: &BasicCard, trump: Suit) -> bool {
        card.suit == trump || self.min_nontrump_rank_to_win.map(|c| card.rank.ord_ace_high() >= c).unwrap_or(false)
    }
}

impl Player for BasicPlayer {
    fn play_card(&self, view: &PlayerView) -> BasicCard {
        let mut cards = view.playable_cards();

        cards.sort_by_key(|c| (c.rank.ord_ace_high(), view.ord_suit(c.suit)));

        match &view.revealed {
            // playing for cards
            &Some(ref c)  => {

                // go all-out for trumps, kings, or better
                if self.try_to_win(c, view.trump) {
                    // Play the highest non-trump, otherwise play the lowest trump
                    cards.iter().rev().find(|p| p.suit != view.trump)
                        .unwrap_or(&cards[0]).clone()
                } else {
                    // try to ditch
                    cards.iter().find(|p| p.suit != view.trump).unwrap_or(&cards[0]).clone()
                }
            },

            // playing for points
            &None => {
                match &view.leading_card {
                    &Some(ref lc) => {
                        // play the lowest card to beat it, otherwise ditch
                        cards.iter().find(|p| view.wins_against(lc, p))
                            .unwrap_or(&cards[0]).clone()
                    },
                    &None => { cards[0].clone() }
                }
            }
        }
    }
}

/// Randomly choose actions at each play
fn play_random_game(start: usize, r: Option<Rank>, verbose: bool) -> Result<[usize; 2], ActionError> {
    let mut round = Round::new(start, (0, 1));

    if verbose {
        println!("{}", format_round(&round));
    }

    let mut actions = round.possible_actions();

    let players: [Box<Player>; 2] = [Box::new(BasicPlayer::new(r)), Box::new(RandomPlayer::new())];

    while actions.len() > 0 {
        let action = {
            let player_view = round.active_player_view();
            let card = players[player_view.player].play_card(&player_view);

            Action { card, player: player_view.player }
        };

        if verbose {
            println!("{}", format_action(&action));
        }

        round.play_action(action)?;

        if verbose && round.get_state().played.is_none() {
            println!("**************");
            println!("{}", format_round(&round));
        }

        actions = round.possible_actions();
    }

    Ok(round.get_state().score.clone())
}

#[allow(unused)]
fn test_basic_player(r: Option<Rank>) -> [usize; 2] {
    let mut games_won = [0, 0];
    let mut starting_player: usize = 0;

    for _ in 0..10 {
        let score = play_random_game(starting_player, r, false).expect("bad game");
        let winner: usize = if score[0] > score[1]  { 0 } else { 1 };
        games_won[winner] += 1;
        starting_player = 1 - winner;
    }

    games_won
}

fn main() {
    cards::auto_suit_colors();

    for r in Rank::iterator() {
        let games_won = test_basic_player(Some(r.clone()));

        println!("Player 1 Record, Rank {}: {}-{}", r, games_won[0], games_won[1]);
    }

    let games_won = test_basic_player(None);

    println!("Player 1 Record, Never: {}-{}", games_won[0], games_won[1]);
    play_random_game(0, Some(Rank::Ace), true);

    let mut b = PlayerBelief::new();
    b.random_cards_drawn(13);
    b.empty_suit(Suit::Clubs);
    b.card_drawn(&"2♦".parse().unwrap());
    b.print_probabilities();
    b.random_cards_drawn(1);
    b.print_probabilities();

}


