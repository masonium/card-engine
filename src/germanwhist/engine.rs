use cards::{BasicCard, Suit};

use super::state::{GameState, PlayerView};
use super::phase::{GamePhase, PlayingPhase, GameOverPhase};

#[derive(Clone, Debug)]
pub struct Action {
    pub player: usize,
    pub card: BasicCard
}

pub type ActionEvent = Action;

#[derive(Clone, Debug)]
pub struct TrickEvent {
    /// who led
    pub leading_player: usize,

    /// who starts the new round
    pub active_player: usize,

    /// cards played during the trick, indexed by player
    pub cards_played: [BasicCard; 2],

    /// new revealed card
    pub revealed: Option<BasicCard>,

    /// score after the trick
    pub score: [usize; 2]
}

#[derive(Clone, Debug)]
pub struct StartRoundEvent {
    pub hand: Vec<BasicCard>,
    pub revealed: BasicCard,
    pub trump: Suit,
    pub starting_player: usize
}

#[derive(Clone, Copy, Debug)]
pub enum ActionError {
    WrongPlayer(usize),
    MissingCard,
    NotFollowingSuit,
    GameOver
}

#[derive(Clone, Copy, Debug)]
pub struct CardEvent {
    pub player: usize,

    /// Card the player received. None indicates the player picked up
    /// a random card.
    pub card: Option<BasicCard>
}

#[derive(Debug, Clone)]
pub enum GameEvent {
    Action(Action),
    Trick(TrickEvent),
    Card(CardEvent),
    Start(StartRoundEvent)
}

pub type ScoringRules = (usize, usize);

/// Game engine for a round of German Whist
pub struct Round {
    state: GameState,
    phase: Option<Box<GamePhase>>,
    rules: ScoringRules,
}

impl Round {
    pub fn new(rules: ScoringRules) -> Round {
        let state = GameState::new(0);
        let phase: Option<Box<GamePhase>> = Some(Box::new(GameOverPhase{}));
        Round { state, phase: phase, rules }
    }

    pub fn start_round(&mut self, starting_player: usize) -> [Vec<GameEvent>; 2] {
        self.phase = Some(Box::new(PlayingPhase{}));
        self.state = GameState::new(starting_player);

        let p0 = StartRoundEvent{ hand: self.state.hands[0].iter().cloned().collect(),
                                  revealed: self.state.revealed.expect("start of round"),
                                  trump: self.state.trump,
                                  starting_player: starting_player };
        let p1 = StartRoundEvent{ hand: self.state.hands[1].iter().cloned().collect(), ..p0 };

        [vec![GameEvent::Start(p0)], vec![GameEvent::Start(p1)]]
    }

    pub fn get_state(&self) -> &GameState {
        &self.state
    }

    pub fn active_player_view(&self) -> PlayerView {
        self.state.player_view(self.state.active)
    }

    pub fn possible_actions(&self) -> Vec<Action> {
        self.phase.as_ref().unwrap().possible_actions(&self.state)
    }

    pub fn get_phase(&self) -> &GamePhase {
        use std::borrow::Borrow;
        self.phase.as_ref().unwrap().borrow()
    }

    pub fn play_action(&mut self, action: Action) -> Result<[Vec<GameEvent>; 2], ActionError> {
        let events = self.phase.as_mut().unwrap().on_action(&mut self.state, &self.rules, action)?;

        if self.state.rounds_left == 0 {
            self.phase = self.phase.as_mut().unwrap().transition(&mut self.state);
        }

        Ok(events)
    }
}
