use termion::color;
use std::fmt;
use rand::{Rng, thread_rng};
use std::slice::Iter;
use atty;

#[derive(Debug, Clone, Copy)]
pub enum ColorMode {
    Plain,
    RedBlack,
    Unique
}

static mut SUIT_COLOR_MODE: ColorMode = ColorMode::Unique;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black,
    Red
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Hearts = 2,
    Spades = 3
}

impl Suit {
    pub fn color(&self) -> Color {
        match *self {
            Suit::Diamonds | Suit::Hearts => Color::Red,
            _ => Color::Black
        }
    }

    pub fn ord(&self) -> u8 {
        *self as u8
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Suit::*;


        lazy_static! {
            static ref RED: String = format!("{}", color::Fg(color::Red));
            static ref GREEN: String = format!("{}", color::Fg(color::Green));
            static ref BLUE: String = format!("{}", color::Fg(color::Blue));
            static ref RESET: String = format!("{}", color::Fg(color::Reset));
        }

        let (begin, end) = unsafe {
            match SUIT_COLOR_MODE {
                ColorMode::Plain => ("", ""),
                ColorMode::RedBlack => match self.color() {
                    Color::Red => (RED.as_str(), RESET.as_str()),
                    _ => ("", "")
                },
                ColorMode::Unique => match *self {
                    Clubs => (GREEN.as_str(), RESET.as_str()),
                    Hearts => (RED.as_str(), RESET.as_str()),
                    Diamonds => (BLUE.as_str(), RESET.as_str()),
                    _ => ("", "")
                }
            }
        };

        write!(f, "{}{}{}", begin, match *self {
            Clubs => "♣",
            Diamonds => "♦",
            Hearts => "♥",
            Spades => "♠"
        }, end)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rank {
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Jack = 11,
    Queen = 12,
    King = 13,
    Ace = 14
}

impl Rank {
    pub fn iterator() -> Iter<'static, Rank> {
        use Rank::*;
        static RANKS: [Rank;  13] = [Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace];
        RANKS.into_iter()
    }
    /// Assign numerical values to each rank, with ace as high
    pub fn ord_ace_high(&self) -> u8 {
        *self as u8
    }

    /// Assign numerical values to each rank, with ace as low
    pub fn ord_ace_low(&self) -> u8 {
        use Rank::*;
        match *self {
            Ace => 1,
            x => x as u8
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Rank::*;
        write!(f, "{}", match *self {
            Two => "2",
            Three => "3",
            Four => "4",
            Five => "5",
            Six => "6",
            Seven => "7",
            Eight => "8",
            Nine => "9",
            Ten => "T",
            Jack => "J",
            Queen => "Q",
            King => "K",
            Ace => "A"
        })
    }
}

/// Normal non-joker playing card
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicCard {
    pub rank: Rank,
    pub suit: Suit
}

impl BasicCard {
    /// Returns a Vec of all 52 possible `BasicCard`s, in
    /// some unspecified order.
    pub fn all() -> Vec<BasicCard> {
        use Rank::*;
        use Suit::*;

        let mut cards = Vec::with_capacity(52);
        for rank in &[Two, Three, Four, Five, Six, Seven, Eight,
                      Nine, Ten, Jack, Queen, King, Ace] {
            for suit in &[Clubs, Diamonds, Hearts, Spades] {
                cards.push(BasicCard { rank: *rank, suit: *suit })
            }
        }

        cards
    }
}

impl fmt::Display for BasicCard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.rank, self.suit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Card {
    Basic(BasicCard),
    BigJoker,
    SmallJoker
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Card::*;
        match self {
            &BigJoker => write!(f, "JJ"),
            &SmallJoker => write!(f, "jj"),
            &Card::Basic(ref basic) => write!(f, "{}", basic)
        }
    }
}

/// Deck of 52 basic (non-joker) cards
#[derive(Debug)]
pub struct BasicDeck {
    cards: Vec<BasicCard>
}

impl BasicDeck {
    pub fn new() -> BasicDeck {
        BasicDeck { cards: BasicCard::all() }
    }

    /// Shuffle the remaining cards in the deck
    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        rng.shuffle(&mut self.cards)
    }

    pub fn num_cards_left(&self) -> usize {
        self.cards.len()
    }

    /// Return the top card from the deck, if there are any cards
    pub fn draw(&mut self) -> Option<BasicCard> {
        self.cards.pop()
    }

    /// Return the top n cards from the deck, if there are n
    /// cards. Otherwise, return None.
    ///
    /// # Remarks
    ///
    /// If this function returns `Some`, there will be
    /// exactly `n` cards.
    pub fn draw_n(&mut self, n: usize) -> Option<Vec<BasicCard>> {
        let m = self.cards.len();
        if m >= n {
            Some(self.cards.split_off(m - n))
        } else {
            None
        }
    }

}

pub fn auto_suit_colors() {
    unsafe {
        SUIT_COLOR_MODE = if atty::is(atty::Stream::Stdout) {
            ColorMode::Unique
        } else {
            ColorMode::Plain
        }
    }
}
