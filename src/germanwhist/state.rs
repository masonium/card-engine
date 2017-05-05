use std::slice;
use std::cmp::Ordering;
use cards::{BasicCard, BasicDeck, Suit};
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
    pub trump: Suit
}

impl GameState {
    /// Create a new round
    pub fn new() -> (GameState, BasicCard) {
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

    /// Return an immutable view of the player's hand.
    pub fn player_view<'a>(&'a self, player: usize) -> PlayerView<'a> {
        PlayerView { hand: &self.hands[player] }
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
    hand: &'a Vec<BasicCard>,
}

impl<'a> PlayerView<'a> {
    pub fn has_card(&self, c: BasicCard) -> bool {
        self.hand.contains(&c)
    }

    pub fn has_suit(&self, s: &Suit) -> bool {
        self.hand.iter().any(|card| card.suit == *s)
    }

    pub fn iter(&self) -> slice::Iter<BasicCard> {
        self.hand.iter()
    }
}
