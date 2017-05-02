use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum Color {
    Black,
    Red
}

#[derive(Debug, Clone, Copy)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades
}

impl Suit {
    pub fn color(&self) -> Color {
        match self {
            Suit::Diamonds | Suit::Hearts => Color::Red,
            _ => Color::Black
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match *self {
            Suit::Clubs => "♣",
            Suit::Diamonds => "♦",
            Suit::Hearts => "♥",
            Suit::Spades => "♠"
        })
    }
}
