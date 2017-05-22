use super::engine::{Action, ScoringRules, ActionError};
use super::state::{GameState};
use super::engine::{CardEvent, TrickEvent, GameEvent};

pub trait GamePhase {
    /// Return the number of possible actions
    fn possible_actions(&self, gs: &GameState) -> Vec<Action>;

    fn is_game_over(&self) -> bool;

    /// perform action on a submitted, return rounds left in this state
    fn on_action(&mut self, gs: &mut GameState, rules: &ScoringRules, action: Action) -> Result<[Vec<GameEvent>; 2], ActionError>;

    fn format(&self, gs: &GameState) -> String;

    /// ending the state
    fn transition(&mut self, gs: &mut GameState) -> Box<GamePhase>;
}

pub struct PlayingPhase;

impl GamePhase for PlayingPhase {
    /// Available actions
    fn possible_actions(&self, gs: &GameState) -> Vec<Action> {
        let view = gs.player_view(gs.active);

        view.playable_cards().into_iter().map(|c| Action {player: gs.active, card: c}).collect()
    }

    fn on_action(&mut self, gs: &mut GameState, rules: &ScoringRules, action: Action) -> Result<[Vec<GameEvent>; 2], ActionError> {
        if action.player != gs.active {
            return Err(ActionError::WrongPlayer(gs.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        let mut events = [Vec::new(), Vec::new()];

        if gs.played.is_none() {
            /// this is the first card played
            gs.player_view_mut(action.player).remove_card(&action.card)?;

            events[0].push(GameEvent::Action(action.clone()));
            events[1].push(GameEvent::Action(action.clone()));

            gs.played = Some(action.card);

            gs.active = 1 - gs.active;
        } else {
            let leading_card = gs.played.take().expect("on_action: already checked !is_none");
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

            let action_ev = GameEvent::Action(action.clone());
            events[0].push(action_ev.clone());
            events[1].push(action_ev.clone());

            let follow = gs.active;
            let lead = 1 - gs.active;

            let winner = if gs.score_hand(&leading_card, &action.card).expect("cards must be distinct") {
                lead
            } else {
                follow
            };
            let loser = 1 - winner;

            let mut cards_played = [leading_card.clone(), action.card.clone()];
            let mut cards_received = [None, None];

            if lead == 1 {
                cards_played.swap(0, 1);
            }

            // hand-building phase
            if gs.revealed.is_some() {
                /// Give players their new cards
                {
                    let r = gs.revealed.take().expect("must be a revealed card");
                    gs.player_view_mut(winner).add_card(r.clone());

                    cards_received[winner] = Some(r);
                    let rec_ev = GameEvent::Card(CardEvent { player: winner, card: Some(r.clone()) });

                    // both players know what the winning player got.
                    events[winner].push(rec_ev.clone());
                    events[loser].push(rec_ev);
                }

                {
                    let draw = gs.draw().expect("must have a card left after trick");
                    gs.player_view_mut(loser).add_card(draw.clone());
                    cards_received[loser] = Some(draw);

                    // the loser knows what they got, but the winner doesn't
                    events[loser].push(GameEvent::Card(CardEvent { player: loser, card: Some(draw.clone()) }));
                    events[winner].push(GameEvent::Card(CardEvent { player: loser, card: None }));
                }

                // Draw a new card, if any
                if gs.deck.num_cards_left() > 0 {
                    gs.revealed = gs.draw();
                }

                gs.increment_score(winner, rules.0);
            }
            // scoring phase
            else {
                gs.increment_score(winner, rules.1);
            }

            gs.active = winner;
            gs.rounds_left -= 1;

            let trick = GameEvent::Trick(TrickEvent{ leading_player: lead,
                                                     active_player: gs.active,
                                                     cards_played,
                                                     revealed: gs.revealed,
                                                     score: gs.score });

            events[loser].push(trick.clone());
            events[winner].push(trick);
        }

        Ok(events)
    }

    fn is_game_over(&self) -> bool {
        false
    }

    fn format(&self, gs: &GameState) -> String {
        match (&gs.revealed, &gs.played) {
            (&None, _) => {
                format!("End of Phase.")
            },
            (&Some(ref reveal), &Some(ref card)) => {
                format!("Playing for {}\nPlayer {} played {}, Player {} to respond.",
                        reveal, 2 - gs.active, card, gs.active + 1)
            },
            (&Some(ref reveal), &None) => {
                format!("Playing for {}\nPlayer {} to open",
                        reveal, gs.active + 1)
            }
        }
    }



    /// Once we're done playing, finish.
    fn transition(&mut self, _: &mut GameState) -> Box<GamePhase> {
        Box::new(GameOverPhase {})
    }

}


pub struct GameOverPhase;

impl GamePhase for GameOverPhase {
    fn possible_actions(&self, _: &GameState) -> Vec<Action> {
        Vec::new()
    }

    fn on_action(&mut self, _: &mut GameState, _: &ScoringRules, _: Action) -> Result<[Vec<GameEvent>; 2], ActionError> {
        Err(ActionError::GameOver)
    }

    fn format(&self, _: &GameState) -> String {
        format!("Game Over.")
    }

    fn is_game_over(&self) -> bool {
        true
    }

    fn transition(&mut self, _: &mut GameState) -> Box<GamePhase> {
        Box::new(GameOverPhase{})
    }
}
