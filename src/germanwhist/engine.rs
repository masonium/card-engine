use cards::{BasicCard};

use super::state::{GameState, PlayerView};
use super::phase::{GamePhase, PlayingPhase};

#[derive(Clone, Debug)]
pub struct Action {
    pub player: usize,
    pub card: BasicCard
}

pub type ActionEvent = Action;

#[derive(Clone, Debug)]
pub struct TrickEvent {
    pub leading_player: usize,
    pub cards: [BasicCard; 2],
    pub score: [usize; 2]
}

#[derive(Clone, Debug)]
pub struct CardEvent {
    pub player: usize,
    pub card: BasicCard
}

#[derive(Clone, Copy, Debug)]
pub enum ActionError {
    WrongPlayer(usize),
    MissingCard,
    NotFollowingSuit,
    GameOver
}

#[derive(Debug)]
pub enum GameEvent {
    Action(Action),
    Trick(TrickEvent),
    Card(CardEvent)
}

pub type ScoringRules = (usize, usize);

/// Game engine for a round of German Whist
pub struct Round {
    state: GameState,
    phase: Option<Box<GamePhase>>,
    rules: ScoringRules,
}

impl Round {
    pub fn new<T: Into<Option<usize>>>(starting_player: T, rules: ScoringRules) -> Round {
        let state = GameState::new(starting_player.into());
        let phase: Option<Box<GamePhase>> = Some(Box::new(PlayingPhase{}));
        Round { state, phase: phase, rules }
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
