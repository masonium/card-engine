use termion::color;
use std::str::FromStr;
use std::fmt;
use rand::{Rng, thread_rng};
use std::slice::Iter;
use std::collections::HashMap;
use atty;

#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum ColorMode {
    Plain = 0,
    RedBlack = 1,
    Unique = 2
}

static mut SUIT_COLOR_MODE: ColorMode = ColorMode::Unique;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black,
    Red
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Hearts = 2,
    Spades = 3
}

static ALL_SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

impl Suit {
    /// iterate through all elements of suit
    pub fn iterator() -> Iter<'static, Suit> {
        ALL_SUITS.into_iter()
    }

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

impl Into<Suit> for u8 {
    fn into(self) -> Suit {
        ALL_SUITS[self as usize]
    }
}

#[derive(Debug)]
pub enum CardParseError {
    BadSuit,
    BadRank
}

impl FromStr for Suit {
    type Err = CardParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use Suit::*;
        use self::CardParseError::*;
        match s {
            "♣" => Ok(Clubs),
            "♦" => Ok(Diamonds),
            "♥" => Ok(Hearts),
            "♠" => Ok(Spades),
            _ => Err(BadSuit)
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    Two = 0,
    Three = 1,
    Four = 2,
    Five = 3,
    Six = 4,
    Seven = 5,
    Eight = 6,
    Nine = 7,
    Ten = 8,
    Jack = 9,
    Queen = 10,
    King = 11,
    Ace = 12
}

static ALL_RANKS: [Rank;  13] = [Rank::Two, Rank::Three, Rank::Four,
                             Rank::Five, Rank::Six, Rank::Seven,
                             Rank::Eight, Rank::Nine, Rank::Ten,
                             Rank::Jack, Rank::Queen, Rank::King, Rank::Ace];

impl Rank {
    pub fn iterator() -> Iter<'static, Rank> {
        ALL_RANKS.into_iter()
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

impl Into<Rank> for u8 {
    fn into(self) -> Rank {
        ALL_RANKS[self as usize]
    }
}

impl FromStr for Rank {
    type Err = CardParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use Rank::*;
        use self::CardParseError::*;
        match s {
            "2" => Ok(Two),
            "3" => Ok(Three),
            "4" => Ok(Four),
            "5" => Ok(Five),
            "6" => Ok(Six),
            "7" => Ok(Seven),
            "8" => Ok(Eight),
            "9" => Ok(Nine),
            "T" => Ok(Ten),
            "J" => Ok(Jack),
            "Q" => Ok(Queen),
            "K" => Ok(King),
            "A" => Ok(Ace),
            _ => Err(BadRank)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasicCard {
    pub rank: Rank,
    pub suit: Suit
}

impl BasicCard {
    /// Returns a Vec of all 52 possible `BasicCard`s, in
    /// some unspecified order.
    pub fn all() -> Vec<BasicCard> {
        use Suit::*;

        let mut cards = Vec::with_capacity(52);
        for rank in Rank::iterator() {
            for suit in &[Clubs, Diamonds, Hearts, Spades] {
                cards.push(BasicCard { rank: *rank, suit: *suit })
            }
        }

        cards
    }
}

impl FromStr for BasicCard {
    type Err = CardParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (rank_str, suit_str) = s.split_at(1);
        let rank = rank_str.parse()?;
        let suit = suit_str.parse()?;
        Ok(BasicCard { rank, suit })
    }
}

impl Into<u8> for BasicCard {
    fn into(self) -> u8 {
        self.rank as u8  + 13 * (self.suit as u8)
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

pub const NUM_BASIC_CARDS: usize = 52;

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

/// Print some value for each card in a hashmap.
pub fn print_card_map<T: fmt::Display>(map: &HashMap<BasicCard, T>) {
    let col_head = "+-----------".to_string().repeat(4);
    println!("{}+", &col_head);

    for rank in Rank::iterator() {
        print!("| ");

        for suit in Suit::iterator() {
            let bc = BasicCard { rank: *rank, suit: *suit };
            print!("{}{}: {:5}", rank, suit, *map.get(&bc).unwrap());
            print!(" | ");
        }
        println!("");
    }
    println!("{}+", &col_head);
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

pub mod prelude {
    pub use super::{BasicCard, Suit, Rank, auto_suit_colors, print_card_map};
}
