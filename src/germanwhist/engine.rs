use cards::{BasicCard, Suit, BasicDeck};
use std::fmt::{self, Display, Formatter};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Action {
    pub player: usize,
    pub card: BasicCard
}

#[derive(Clone, Copy)]
pub enum ActionError {
    WrongPlayer(usize),
    MissingCard,
    NotFollowingSuit,
    GameOver
}

pub trait PhaseState: Display {
    /// perform action on a submitted, return rounds left in this state
    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError>;

    /// ending the state
    fn transition(&mut self, gs: &mut GameState) -> Option<Box<PhaseState>>;
}

pub struct FirstPhase {
    /// card currently played
    played: Option<BasicCard>,

    /// player whose turn it is
    active: usize,

    rounds_left: usize,

    // currently revealed card, if any
    revealed: Option<BasicCard>,
}

impl FirstPhase {
    fn new(first_card: BasicCard, player: usize) -> Self {
        FirstPhase { played: None, active: player,
                     rounds_left: 13,
                     revealed: Some(first_card) }
    }
}

impl Display for FirstPhase {
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

impl PhaseState for FirstPhase {
    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError> {
        if action.player != self.active {
            return Err(ActionError::WrongPlayer(self.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        match self.played {
            /// this is the first card played
            None => {
                gs.player_view_mut(action.player).remove_card(&action.card)?;

                self.played = Some(action.card);
                self.active = 1 - self.active;
            },
            /// second card played; this will finish a trick
            Some(ref card) => {
                let trump = gs.trump;
                {
                    let mut player = gs.player_view_mut(action.player);

                    // If the player has the suit, the card must match
                    if player.has_suit(&trump) {
                        if action.card.suit != trump {
                            return Err(ActionError::NotFollowingSuit);
                        }
                    }

                    player.remove_card(&action.card)?;
                }

                let follow = self.active;
                let lead = 1 - self.active;

                let winner = if gs.score_hand(card, &action.card).unwrap() {
                    lead
                } else {
                    follow
                };
                let loser = 1 - winner;

                /// Give players their new cards
                let r = self.revealed.take().expect("must be a revealed card");
                gs.player_view_mut(winner).add_card(r);

                let draw = gs.draw().expect("must have a card left after trick");
                gs.player_view_mut(loser).add_card(draw);

                // Draw a new card, if any
                if gs.deck.num_cards_left() > 0 {
                    self.revealed = gs.draw();
                }

                self.active = winner;
                self.rounds_left -= 1;
            }

        }

        Ok(self.rounds_left)
    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<PhaseState>> {
        Some(Box::new(SecondPhase::new(self.active)))
    }

}


impl Display for SecondPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.rounds_left == 0 {
            write!(f, "end of phase")
        } else {
            match self.played {
                Some(ref card) => {
                    write!(f, "layer {} played {}, Player {} to respond.",
                           1 - self.active, card, self.active)
                },
                None => {
                    write!(f, "Player {} to open",
                           1 - self.active)
                }
            }
        }
    }
}

pub struct SecondPhase {
    /// card currently played
    played: Option<BasicCard>,

    /// player whose turn it is
    active: usize,

    /// rounds left
    rounds_left: usize
}

impl SecondPhase {
    fn new(player: usize) -> Self {
        SecondPhase{ active: player, played: None, rounds_left: 13 }
    }
}

impl PhaseState for SecondPhase {

    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError> {
        if action.player != self.active {
            return Err(ActionError::WrongPlayer(self.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        match self.played {
            /// this is the first card played
            None => {
                gs.player_view_mut(action.player).remove_card(&action.card)?;

                self.played = Some(action.card);
                self.active = 1 - self.active;
            },
            /// second card played; this will finish a trick
            Some(ref card) => {
                let trump = gs.trump;
                {
                    let mut player = gs.player_view_mut(action.player);

                    // If the player has the suit, the card must match
                    if player.has_suit(&trump) {
                        if action.card.suit != trump {
                            return Err(ActionError::NotFollowingSuit);
                        }
                    }

                    player.remove_card(&action.card)?;
                }

                let follow = self.active;
                let lead = 1 - self.active;

                let winner = if gs.score_hand(card, &action.card).unwrap() {
                    lead
                } else {
                    follow
                };

                // The winner of the trick score a point and plays the next hand.
                gs.increment_score(winner, 1);

                self.active = winner;
                self.rounds_left -= 1;
            }

        }

        Ok(self.rounds_left)

    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<PhaseState>> {
        Some(Box::new(GameOverPhase{}))
    }

}

pub struct GameOverPhase;

impl Display for GameOverPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Game Over.")
    }
}

impl PhaseState  for GameOverPhase {
    fn on_action(&mut self, _: &mut GameState, _: Action) -> Result<usize, ActionError> {
        Err(ActionError::GameOver)
    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<PhaseState>> {
        Some(Box::new(GameOverPhase{}))
    }
}

#[derive(Debug)]
pub struct GameState {
    /// current deck
    pub deck: BasicDeck,

    /// hands for each player
    pub hands: [Vec<BasicCard>; 2],

    /// score for each player
    pub score: [usize; 2],

    /// trump suit
    pub trump: Suit
}

impl GameState {
    /// Create a new round
    fn new() -> (GameState, BasicCard) {
        let mut deck = BasicDeck::new();
        deck.shuffle();


        let hands = [deck.draw_n(13).unwrap(),
                     deck.draw_n(13).unwrap()];
        let c = deck.draw().unwrap();
        let trump = c.suit;
        let score = [0, 0];

        (GameState { deck, hands, score, trump }, c)
    }

    /// Return a mutable view of the player's hand.
    pub fn player_view_mut<'a>(&'a mut self, player: usize) -> PlayerViewMut<'a> {
        PlayerViewMut { hand: &mut self.hands[player] }
    }

    /// Return true iff the leading player wins the trick
    pub fn score_hand(&self, leading: &BasicCard, following: &BasicCard) -> Option<bool> {
        if leading == following {
            return None;
        }

        if leading.suit == self.trump {
            Some(following.suit != self.trump || leading.rank.ord_ace_high() > following.rank.ord_ace_high())
        } else {
            Some(if following.suit == self.trump {
                false
            } else if following.suit != leading.suit {
                true
            } else {
                leading.rank.ord_ace_high() > following.rank.ord_ace_high()
            })
        }
    }

    /// Reveal a new top card
    pub fn draw(&mut self) -> Option<BasicCard> {
        self.deck.draw()
    }

    /// Add score to specified player
    pub fn increment_score(&mut self, player: usize, points: usize) {
        self.score[player] += points;
    }

    /// Utility function to display cards in order.
    ///
    /// Group by suit, trumps first, ordered within suit, ace_high
    pub fn display_order(&self, c1: &BasicCard, c2: &BasicCard) -> Ordering {
        let s1 = c1.suit.ord() + if  c1.suit != self.trump { 4 } else { 0 };
        let s2 = c2.suit.ord() + if  c2.suit != self.trump { 4 } else { 0 };
        (s1, c1.rank.ord_ace_high()).cmp(&(s2, c2.rank.ord_ace_high()))
    }
}

pub struct PlayerViewMut<'a> {
    hand: &'a mut Vec<BasicCard>,
}

impl<'a> PlayerViewMut<'a> {
    pub fn has_card(&self, c: BasicCard) -> bool {
        self.hand.contains(&c)
    }

    pub fn has_suit(&self, s: &Suit) -> bool {
        self.hand.iter().any(|card| card.suit == *s)
    }

    pub fn add_card(&mut self, c: BasicCard) {
        self.hand.push(c)
    }

    /// Remove the specified card from the players hand, returning an
    /// error if the player didn't own the card.
    pub fn remove_card(&mut self, c: &BasicCard) -> Result<(), ActionError> {
        let p = self.hand.iter().position(|x| x == c);
        match p  {
            Some(i) => {
                self.hand.remove(i);
                Ok(())
            },
            None => {
                Err(ActionError::MissingCard)
            }
        }
    }
}

pub struct Round {
    state: GameState,

    phase: Option<Box<PhaseState>>
}

impl Round {
    pub fn new() -> Round {
        let (state, card) = GameState::new();
        let phase: Option<Box<PhaseState>> = Some(Box::new(FirstPhase::new(card, 0)));
        Round { state, phase: phase }
    }

    pub fn get_state(&self) -> &GameState {
        &self.state
    }

    pub fn get_phase(&self) -> &PhaseState {
        use std::borrow::Borrow;
        self.phase.as_ref().unwrap().borrow()
    }

    pub fn play_action(&mut self, action: Action) -> Result<(), ActionError> {
        let rl = self.phase.as_mut().unwrap().on_action(&mut self.state, action)?;

        if rl == 0 {
            self.phase = self.phase.as_mut().unwrap().transition(&mut self.state);
        }

        Ok(())
    }
}
