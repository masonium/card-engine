use std::slice;
use std::cmp::Ordering;
use crate::cards::{BasicCard, BasicDeck, Suit};
use super::engine::{ActionError};

#[derive(Debug)]
pub struct GameState {
    /// current deck
    pub deck: BasicDeck,

    /// hands for each player
    pub hands: [Vec<BasicCard>; 2],

    /// score for each player
    pub score: [usize; 2],

    /// trump suit
    pub trump: Suit,

    /// card currently played
    pub played: Option<BasicCard>,

    /// player whose turn it is
    pub active: usize,

    /// number of rounds left in this phase
    pub rounds_left: usize,

    // currently revealed card, if any
    pub revealed: Option<BasicCard>,

}

impl GameState {
    /// Create a new round
    pub fn new<T: Into<Option<usize>>>(player: T) -> GameState {
        let mut deck = BasicDeck::new();
        deck.shuffle();

        let hands = [deck.draw_n(13).unwrap(),
                     deck.draw_n(13).unwrap()];
        let c = deck.draw().expect("deck has 26 cards left");
        let trump = c.suit;
        let score = [0, 0];
        let active: usize = player.into().unwrap_or(0);
        let played = None;
        let rounds_left = 26;

        GameState { deck, hands, score, trump,
                    active, played, rounds_left, revealed: Some(c) }
    }

    /// Return a mutable view of the player's hand.
    pub fn player_view_mut<'a>(&'a mut self, player: usize) -> PlayerViewMut<'a> {
        PlayerViewMut { hand: &mut self.hands[player] }
    }

    /// Return an immutable view of the player's hand.
    pub fn player_view<'a>(&'a self, player: usize) -> PlayerView<'a> {
        PlayerView::from_state(player, &self)
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

pub struct PlayerView<'a> {
    /// player's current hand
    hand: &'a [BasicCard],

    pub player: usize,

    pub revealed: Option<BasicCard>,
    pub leading_card: Option<BasicCard>,

    pub trump: Suit,

    pub score: [usize; 2]
}

impl<'a> PlayerView<'a> {
    pub fn from_state(player: usize, gs: &GameState) -> PlayerView {

        PlayerView { player,
                     hand: &gs.hands[player],
                     revealed: gs.revealed.clone(),
                     leading_card: gs.played.clone(),
                     trump: gs.trump,
                     score: gs.score.clone() }
    }

    /// Return the set of cards playable in the current state.
    ///
    /// Assumes the player is active.
    pub fn playable_cards(&self) -> Vec<BasicCard> {
        match &self.leading_card {
            // Second player must follow suit, if possible.
            &Some(ref c) if self.has_suit(&c.suit) =>
                self.hand.iter().filter(|x| x.suit == c.suit)
                .cloned().collect()
                ,
            // Otherwise, can play anything
            _ => self.hand.iter().cloned().collect()
        }
    }

    pub fn has_card(&self, c: BasicCard) -> bool {
        self.hand.contains(&c)
    }

    pub fn has_suit(&self, s: &Suit) -> bool {
        self.hand.iter().any(|card| card.suit == *s)
    }

    pub fn ord_suit(&self, s: Suit) -> u8 {
        s.ord() + if s == self.trump { 4 } else { 0 }
    }

    /// Return true if the following card  beats the leading card.
    ///
    /// When cards are equivalent, return false.
    pub fn wins_against(&self, leading: &BasicCard, follow: &BasicCard) -> bool {
        if follow.suit == leading.suit {
            follow.rank.ord_ace_high() > leading.rank.ord_ace_high()
        } else {
            follow.suit == self.trump
        }
    }

    pub fn iter(&self) -> slice::Iter<BasicCard> {
        self.hand.iter()
    }
}
