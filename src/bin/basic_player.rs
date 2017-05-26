extern crate card_engine;
extern crate rand;
extern crate ndarray;
extern crate time;

use card_engine::{GameEvent, Round, Action, ActionError};
use card_engine::cards::{self, BasicCard, Rank, Suit};
use card_engine::germanwhist::util::*;
use card_engine::germanwhist::{PlayerView, PlayerState};
use card_engine::learning::training::{SarsaPlayer, SarsaLambda, SarsaLambdaParameters};
use card_engine::learning::neural_net::{NeuralNet, LayerDesc, ActivationFunction};
use ndarray::Array;
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
#[allow(unused)]
fn play_random_game(start: usize, r: Option<Rank>, verbose: bool) -> Result<[usize; 2], ActionError> {
    let mut round = Round::new((0, 1));

    let mut ps = PlayerState::new(0);

    let players: [Box<Player>; 2] = [Box::new(BasicPlayer::new(r)), Box::new(RandomPlayer::new())];

    let events = round.start_round(start);
    for ev in &events[0] {
        ps.on_event(ev);
    }
    let mut actions = round.possible_actions();

    let mut iter = 1;
    while actions.len() > 0 {
        println!("+++++ Round: {} +++++", iter);
        iter += 1;
        println!("{}", ps);
        let action = {
            let player_view = round.active_player_view();
            let card = players[player_view.player].play_card(&player_view);

            Action { card, player: player_view.player }
        };

        if verbose {
            println!("{}", format_action(&action));
        }

        let events = round.play_action(action)?;
        for ev in &events[0] {
            ps.on_event(ev);
        }

        assert!(ps.oppo.matches_hand(round.get_state().hands[1].iter()));

        if verbose && round.get_state().played.is_none() {
            println!("**************");
            println!("{}", format_round(&round));
        }

        actions = round.possible_actions();
    }
    println!("{}", ps);

    Ok(round.get_state().score.clone())
}

fn basic_random_game() -> f32 {
    let mut round = Round::new((0, 1));
    let mut ps = PlayerState::new(0);

    let players: [Box<Player>; 2] = [Box::new(BasicPlayer::new(None)), Box::new(RandomPlayer::new())];

    let events = round.start_round(None);
    for ev in &events[0] {
        ps.on_event(ev);
    }
    let mut actions = round.possible_actions();

    while actions.len() > 0 {
        let action = {
            let player_view = round.active_player_view();
            let card = players[player_view.player].play_card(&player_view);

            Action { card, player: player_view.player }
        };

        let events = round.play_action(action).unwrap();
        for ev in &events[0] {
            ps.on_event(ev);
        }

        actions = round.possible_actions();
    }

    let s = round.get_state().score;
    if (s[0] as f32 - s[1] as f32) > 0.0 { 1.0 } else { 0.0 }
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

fn test_against<P: Player>(nn: &NeuralNet, oppo: P) -> f32 {
    let mut eng = Round::new((0, 1));

    let mut ps = SarsaPlayer::new(PlayerState::new(1), nn.num_parameters());

    let evs = eng.start_round(None);
    for ev in &evs[1] {
        ps.state.on_event(ev);
    }

    /// start the game
    let mut actions = eng.possible_actions();
    let mut sa = Array::zeros(ps.state.state_vector_size() + ps.state.action_vector_size());

    while actions.len() > 0 {

        let action = if eng.active_player() == 0 {
            let c = oppo.play_card(&eng.active_player_view());
            Action { player: 0, card: c }
        } else {
            ps.greedy_action(nn, &actions, sa.view_mut())
        };

        let evs = eng.play_action(action).expect("must play valid action");
        
        for ev in &evs[1] {
            ps.state.on_event(ev);
        }

        actions = eng.possible_actions();
    }

    if eng.get_state().score[1] as f32 - eng.get_state().score[0] as f32 > 0.0 { 1.0 } else { 0.0 }
}

fn main() {
    cards::auto_suit_colors();
    let sa = PlayerState::action_size() + PlayerState::state_size();
    let nn = NeuralNet::new(&[LayerDesc::new(sa, 100, ActivationFunction::SymmetricSigmoid),
                              LayerDesc::new(100, 1, ActivationFunction::Sigmoid)],
                            0.05).unwrap();

    let mut sl = SarsaLambda::new((0, 1), nn, SarsaLambdaParameters::default())
        .ok().expect("sarsa lambda creation");

    let mut basic_sd: f32 = 0.0;
    let mut random_sd: f32 = 0.0;
    let mut br_sd: f32 = 0.0;
    let decay: f32 = 0.95;
    for i in 0..1000000 {
        if i % 10 == 0 {
            basic_sd = basic_sd * decay + test_against(sl.current_model(), BasicPlayer::new(None)) * (1.0 - decay);
            random_sd = random_sd * decay + test_against(sl.current_model(), RandomPlayer) * (1.0 - decay);
            br_sd = br_sd * decay + basic_random_game() * (1.0 - decay);
        }
        if i % 100 == 0 {
            println!("{:10}, {}, {}, {}", i, basic_sd, random_sd, br_sd);
        }
        if i % 1000 == 0 {
            println!("{}", time::now().strftime("%H:%M:%S").ok().unwrap());
        }
        sl.train_on_episode(false);
    }

    //play_random_game(0, Some(Rank::Ace), true);
}


