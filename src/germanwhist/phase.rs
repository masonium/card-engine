use std::fmt::{self, Display, Formatter};
use super::engine::{Action, ActionError};
use super::state::{GameState};
use cards::{BasicCard};

pub trait GamePhase: Display {
    fn possible_actions(&self, gs: &GameState) -> Vec<Action>;

    /// perform action on a submitted, return rounds left in this state
    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError>;

    /// ending the state
    fn transition(&mut self, gs: &mut GameState) -> Option<Box<GamePhase>>;
}

pub struct PlayingPhase {
    /// card currently played
    played: Option<BasicCard>,

    /// player whose turn it is
    active: usize,

    /// number of rounds left in this phase
    rounds_left: usize,

    // currently revealed card, if any
    revealed: Option<BasicCard>,
}

impl PlayingPhase {
    pub fn new(first_card: BasicCard, player: usize) -> Self {
        PlayingPhase { played: None, active: player,
                       rounds_left: 26,
                       revealed: Some(first_card) }
    }
}

impl Display for PlayingPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match (&self.revealed, &self.played) {
            (&None, _) => {
                write!(f, "End of Phase.")
            },
            (&Some(ref reveal), &Some(ref card)) => {
                write!(f, "Playing for {}\nPlayer {} played {}, Player {} to respond.",
                       reveal, 2 - self.active, card, self.active + 1)
            },
            (&Some(ref reveal), &None) => {
                write!(f, "Playing for {}\nPlayer {} to open",
                       reveal, self.active + 1)
            }
        }
    }
}

impl GamePhase for PlayingPhase {
    /// Available actions
    fn possible_actions(&self, gs: &GameState) -> Vec<Action> {
        let view = gs.player_view(self.active);

        match &self.played {
            // Second player must follow suit, if possible.
            &Some(ref c) if view.has_suit(&c.suit) =>
                view.iter().filter(|x| x.suit == c.suit)
                .map(|c| Action { player: self.active, card: c.clone()}).collect()
                ,
            // Otherwise, can play anything
            _ => view.iter().map(|c| Action { player: self.active, card: c.clone()}).collect(),
        }
    }

    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError> {
        if action.player != self.active {
            return Err(ActionError::WrongPlayer(self.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        if self.played.is_none() {
            /// this is the first card played
            gs.player_view_mut(action.player).remove_card(&action.card)?;

            self.played = Some(action.card);
            self.active = 1 - self.active;
        } else {
            let leading_card = self.played.take().expect("on_action: already checked !is_none");
            {
                let mut player = gs.player_view_mut(action.player);

                // If the player has the suit, the card must match
                if player.has_suit(&leading_card.suit) {
                    if action.card.suit != leading_card.suit {
                        return Err(ActionError::NotFollowingSuit);
                    }
                }

                player.remove_card(&action.card)?;
            }

            let follow = self.active;
            let lead = 1 - self.active;

            let winner = if gs.score_hand(&leading_card, &action.card).expect("cards must be distinct") {
                lead
            } else {
                follow
            };
            let loser = 1 - winner;

            // hand-building phase
            if self.revealed.is_some() {
                /// Give players their new cards
                let r = self.revealed.take().expect("must be a revealed card");
                gs.player_view_mut(winner).add_card(r);

                let draw = gs.draw().expect("must have a card left after trick");
                gs.player_view_mut(loser).add_card(draw);

                // Draw a new card, if any
                if gs.deck.num_cards_left() > 0 {
                    self.revealed = gs.draw();
                }
            }
            // scoring phase
            else {
                gs.increment_score(winner, 1);
            }

            self.active = winner;
            self.rounds_left -= 1;
        }

        Ok(self.rounds_left)
    }

    /// Once we're done playing, finish.
    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(GameOverPhase {}))
    }

}


pub struct GameOverPhase;

impl Display for GameOverPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Game Over.")
    }
}

impl GamePhase for GameOverPhase {
    fn possible_actions(&self, _: &GameState) -> Vec<Action> {
        Vec::new()
    }

    fn on_action(&mut self, _: &mut GameState, _: Action) -> Result<usize, ActionError> {
        Err(ActionError::GameOver)
    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(GameOverPhase{}))
    }
}
