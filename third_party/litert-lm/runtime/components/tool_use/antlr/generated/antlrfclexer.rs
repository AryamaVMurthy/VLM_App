// Generated from AntlrFcLexer.g4 by ANTLR 4.13.2
#![allow(dead_code)]
#![allow(nonstandard_style)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use antlr4rust::atn::ATN;
use antlr4rust::atn_deserializer::ATNDeserializer;
use antlr4rust::char_stream::CharStream;
use antlr4rust::dfa::DFA;
use antlr4rust::error_listener::ErrorListener;
use antlr4rust::int_stream::IntStream;
use antlr4rust::lexer::{BaseLexer, Lexer, LexerRecog};
use antlr4rust::lexer_atn_simulator::{ILexerATNSimulator, LexerATNSimulator};
use antlr4rust::parser_rule_context::{cast, BaseParserRuleContext, ParserRuleContext};
use antlr4rust::recognizer::{Actions, Recognizer};
use antlr4rust::rule_context::{BaseRuleContext, EmptyContext, EmptyCustomRuleContext};
use antlr4rust::token::*;
use antlr4rust::token_factory::{CommonTokenFactory, TokenAware, TokenFactory};
use antlr4rust::tree::ParseTree;
use antlr4rust::vocabulary::{Vocabulary, VocabularyImpl};
use antlr4rust::PredictionContextCache;
use antlr4rust::TokenSource;

use antlr4rust::{lazy_static, Tid, TidAble, TidExt};

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

pub const OPEN_BRACE: i32 = 1;
pub const CLOSE_BRACE: i32 = 2;
pub const OPEN_BRACKET: i32 = 3;
pub const CLOSE_BRACKET: i32 = 4;
pub const COMMA: i32 = 5;
pub const COLON: i32 = 6;
pub const ESCAPE: i32 = 7;
pub const BOOLEAN: i32 = 8;
pub const NULL_LITERAL: i32 = 9;
pub const NUMBER: i32 = 10;
pub const ESCAPED_STRING: i32 = 11;
pub const CALL: i32 = 12;
pub const ID: i32 = 13;
pub const WS: i32 = 14;
pub const channelNames: [&'static str; 0 + 2] = ["DEFAULT_TOKEN_CHANNEL", "HIDDEN"];

pub const modeNames: [&'static str; 1] = ["DEFAULT_MODE"];

pub const ruleNames: [&'static str; 17] = [
    "OPEN_BRACE",
    "CLOSE_BRACE",
    "OPEN_BRACKET",
    "CLOSE_BRACKET",
    "COMMA",
    "COLON",
    "ESCAPE",
    "BOOLEAN",
    "NULL_LITERAL",
    "NUMBER",
    "INT",
    "FRAC",
    "EXP",
    "ESCAPED_STRING",
    "CALL",
    "ID",
    "WS",
];

pub const _LITERAL_NAMES: [Option<&'static str>; 13] = [
    None,
    Some("'{'"),
    Some("'}'"),
    Some("'['"),
    Some("']'"),
    Some("','"),
    Some("':'"),
    None,
    None,
    Some("'null'"),
    None,
    None,
    Some("'call'"),
];
pub const _SYMBOLIC_NAMES: [Option<&'static str>; 15] = [
    None,
    Some("OPEN_BRACE"),
    Some("CLOSE_BRACE"),
    Some("OPEN_BRACKET"),
    Some("CLOSE_BRACKET"),
    Some("COMMA"),
    Some("COLON"),
    Some("ESCAPE"),
    Some("BOOLEAN"),
    Some("NULL_LITERAL"),
    Some("NUMBER"),
    Some("ESCAPED_STRING"),
    Some("CALL"),
    Some("ID"),
    Some("WS"),
];
lazy_static! {
    static ref _shared_context_cache: Arc<PredictionContextCache> =
        Arc::new(PredictionContextCache::new());
    static ref VOCABULARY: Box<dyn Vocabulary> = Box::new(VocabularyImpl::new(
        _LITERAL_NAMES.iter(),
        _SYMBOLIC_NAMES.iter(),
        None
    ));
}

pub type LexerContext<'input> =
    BaseRuleContext<'input, EmptyCustomRuleContext<'input, LocalTokenFactory<'input>>>;
pub type LocalTokenFactory<'input> = CommonTokenFactory;

type From<'a> = <LocalTokenFactory<'a> as TokenFactory<'a>>::From;

pub struct AntlrFcLexer<'input, Input: CharStream<From<'input>>> {
    base: BaseLexer<'input, AntlrFcLexerActions, Input, LocalTokenFactory<'input>>,
}

antlr4rust::tid! { impl<'input,Input> TidAble<'input> for AntlrFcLexer<'input,Input> where Input:CharStream<From<'input> > }

impl<'input, Input: CharStream<From<'input>>> Deref for AntlrFcLexer<'input, Input> {
    type Target = BaseLexer<'input, AntlrFcLexerActions, Input, LocalTokenFactory<'input>>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'input, Input: CharStream<From<'input>>> DerefMut for AntlrFcLexer<'input, Input> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<'input, Input: CharStream<From<'input>>> AntlrFcLexer<'input, Input> {
    fn get_rule_names(&self) -> &'static [&'static str] {
        &ruleNames
    }
    fn get_literal_names(&self) -> &[Option<&str>] {
        &_LITERAL_NAMES
    }

    fn get_symbolic_names(&self) -> &[Option<&str>] {
        &_SYMBOLIC_NAMES
    }

    fn get_grammar_file_name(&self) -> &'static str {
        "AntlrFcLexer.g4"
    }

    pub fn new_with_token_factory(input: Input, tf: &'input LocalTokenFactory<'input>) -> Self {
        antlr4rust::recognizer::check_version("0", "5");
        Self {
            base: BaseLexer::new_base_lexer(
                input,
                LexerATNSimulator::new_lexer_atnsimulator(
                    _ATN.clone(),
                    _decision_to_DFA.clone(),
                    _shared_context_cache.clone(),
                ),
                AntlrFcLexerActions {},
                tf,
            ),
        }
    }
}

impl<'input, Input: CharStream<From<'input>>> AntlrFcLexer<'input, Input>
where
    &'input LocalTokenFactory<'input>: Default,
{
    pub fn new(input: Input) -> Self {
        AntlrFcLexer::new_with_token_factory(
            input,
            <&LocalTokenFactory<'input> as Default>::default(),
        )
    }
}

pub struct AntlrFcLexerActions {}

impl AntlrFcLexerActions {}

impl<'input, Input: CharStream<From<'input>>>
    Actions<'input, BaseLexer<'input, AntlrFcLexerActions, Input, LocalTokenFactory<'input>>>
    for AntlrFcLexerActions
{
}

impl<'input, Input: CharStream<From<'input>>> AntlrFcLexer<'input, Input> {}

impl<'input, Input: CharStream<From<'input>>>
    LexerRecog<'input, BaseLexer<'input, AntlrFcLexerActions, Input, LocalTokenFactory<'input>>>
    for AntlrFcLexerActions
{
}
impl<'input> TokenAware<'input> for AntlrFcLexerActions {
    type TF = LocalTokenFactory<'input>;
}

impl<'input, Input: CharStream<From<'input>>> TokenSource<'input> for AntlrFcLexer<'input, Input> {
    type TF = LocalTokenFactory<'input>;

    fn next_token(&mut self) -> <Self::TF as TokenFactory<'input>>::Tok {
        self.base.next_token()
    }

    fn get_line(&self) -> isize {
        self.base.get_line()
    }

    fn get_char_position_in_line(&self) -> isize {
        self.base.get_char_position_in_line()
    }

    fn get_input_stream(&mut self) -> Option<&mut dyn IntStream> {
        self.base.get_input_stream()
    }

    fn get_source_name(&self) -> String {
        self.base.get_source_name()
    }

    fn get_token_factory(&self) -> &'input Self::TF {
        self.base.get_token_factory()
    }

    fn get_dfa_string(&self) -> String {
        self.base.get_dfa_string()
    }
}

lazy_static! {
    static ref _ATN: Arc<ATN> =
        Arc::new(ATNDeserializer::new(None).deserialize(&mut _serializedATN.iter()));
    static ref _decision_to_DFA: Arc<Vec<antlr4rust::RwLock<DFA>>> = {
        let mut dfa = Vec::new();
        let size = _ATN.decision_to_state.len() as i32;
        for i in 0..size {
            dfa.push(DFA::new(_ATN.clone(), _ATN.get_decision_state(i), i).into())
        }
        Arc::new(dfa)
    };
    static ref _serializedATN: Vec<i32> = vec![
        4, 0, 14, 152, 6, -1, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 4, 2, 5, 7,
        5, 2, 6, 7, 6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2, 10, 7, 10, 2, 11, 7, 11, 2, 12, 7,
        12, 2, 13, 7, 13, 2, 14, 7, 14, 2, 15, 7, 15, 2, 16, 7, 16, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2,
        1, 2, 1, 3, 1, 3, 1, 4, 1, 4, 1, 5, 1, 5, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6,
        1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 3, 6, 64, 8, 6, 1, 7, 1, 7, 1, 7, 1, 7, 1,
        7, 1, 7, 1, 7, 1, 7, 1, 7, 3, 7, 75, 8, 7, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 9, 3, 9, 83, 8,
        9, 1, 9, 1, 9, 1, 9, 3, 9, 88, 8, 9, 1, 9, 3, 9, 91, 8, 9, 1, 9, 1, 9, 3, 9, 95, 8, 9, 1,
        9, 3, 9, 98, 8, 9, 1, 10, 1, 10, 1, 10, 5, 10, 103, 8, 10, 10, 10, 12, 10, 106, 9, 10, 3,
        10, 108, 8, 10, 1, 11, 1, 11, 4, 11, 112, 8, 11, 11, 11, 12, 11, 113, 1, 12, 1, 12, 3, 12,
        118, 8, 12, 1, 12, 4, 12, 121, 8, 12, 11, 12, 12, 12, 122, 1, 13, 1, 13, 5, 13, 127, 8, 13,
        10, 13, 12, 13, 130, 9, 13, 1, 13, 1, 13, 1, 14, 1, 14, 1, 14, 1, 14, 1, 14, 1, 15, 1, 15,
        5, 15, 141, 8, 15, 10, 15, 12, 15, 144, 9, 15, 1, 16, 4, 16, 147, 8, 16, 11, 16, 12, 16,
        148, 1, 16, 1, 16, 1, 128, 0, 17, 1, 1, 3, 2, 5, 3, 7, 4, 9, 5, 11, 6, 13, 7, 15, 8, 17, 9,
        19, 10, 21, 0, 23, 0, 25, 0, 27, 11, 29, 12, 31, 13, 33, 14, 1, 0, 7, 1, 0, 49, 57, 1, 0,
        48, 57, 2, 0, 69, 69, 101, 101, 2, 0, 43, 43, 45, 45, 3, 0, 65, 90, 95, 95, 97, 122, 4, 0,
        48, 57, 65, 90, 95, 95, 97, 122, 3, 0, 9, 10, 13, 13, 32, 32, 165, 0, 1, 1, 0, 0, 0, 0, 3,
        1, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0, 9, 1, 0, 0, 0, 0, 11, 1, 0, 0, 0, 0, 13,
        1, 0, 0, 0, 0, 15, 1, 0, 0, 0, 0, 17, 1, 0, 0, 0, 0, 19, 1, 0, 0, 0, 0, 27, 1, 0, 0, 0, 0,
        29, 1, 0, 0, 0, 0, 31, 1, 0, 0, 0, 0, 33, 1, 0, 0, 0, 1, 35, 1, 0, 0, 0, 3, 37, 1, 0, 0, 0,
        5, 39, 1, 0, 0, 0, 7, 41, 1, 0, 0, 0, 9, 43, 1, 0, 0, 0, 11, 45, 1, 0, 0, 0, 13, 63, 1, 0,
        0, 0, 15, 74, 1, 0, 0, 0, 17, 76, 1, 0, 0, 0, 19, 97, 1, 0, 0, 0, 21, 107, 1, 0, 0, 0, 23,
        109, 1, 0, 0, 0, 25, 115, 1, 0, 0, 0, 27, 124, 1, 0, 0, 0, 29, 133, 1, 0, 0, 0, 31, 138, 1,
        0, 0, 0, 33, 146, 1, 0, 0, 0, 35, 36, 5, 123, 0, 0, 36, 2, 1, 0, 0, 0, 37, 38, 5, 125, 0,
        0, 38, 4, 1, 0, 0, 0, 39, 40, 5, 91, 0, 0, 40, 6, 1, 0, 0, 0, 41, 42, 5, 93, 0, 0, 42, 8,
        1, 0, 0, 0, 43, 44, 5, 44, 0, 0, 44, 10, 1, 0, 0, 0, 45, 46, 5, 58, 0, 0, 46, 12, 1, 0, 0,
        0, 47, 48, 5, 60, 0, 0, 48, 49, 5, 101, 0, 0, 49, 50, 5, 115, 0, 0, 50, 51, 5, 99, 0, 0,
        51, 52, 5, 97, 0, 0, 52, 53, 5, 112, 0, 0, 53, 54, 5, 101, 0, 0, 54, 64, 5, 62, 0, 0, 55,
        56, 5, 60, 0, 0, 56, 57, 5, 99, 0, 0, 57, 58, 5, 116, 0, 0, 58, 59, 5, 114, 0, 0, 59, 60,
        5, 108, 0, 0, 60, 61, 5, 52, 0, 0, 61, 62, 5, 54, 0, 0, 62, 64, 5, 62, 0, 0, 63, 47, 1, 0,
        0, 0, 63, 55, 1, 0, 0, 0, 64, 14, 1, 0, 0, 0, 65, 66, 5, 116, 0, 0, 66, 67, 5, 114, 0, 0,
        67, 68, 5, 117, 0, 0, 68, 75, 5, 101, 0, 0, 69, 70, 5, 102, 0, 0, 70, 71, 5, 97, 0, 0, 71,
        72, 5, 108, 0, 0, 72, 73, 5, 115, 0, 0, 73, 75, 5, 101, 0, 0, 74, 65, 1, 0, 0, 0, 74, 69,
        1, 0, 0, 0, 75, 16, 1, 0, 0, 0, 76, 77, 5, 110, 0, 0, 77, 78, 5, 117, 0, 0, 78, 79, 5, 108,
        0, 0, 79, 80, 5, 108, 0, 0, 80, 18, 1, 0, 0, 0, 81, 83, 5, 45, 0, 0, 82, 81, 1, 0, 0, 0,
        82, 83, 1, 0, 0, 0, 83, 84, 1, 0, 0, 0, 84, 87, 3, 21, 10, 0, 85, 88, 3, 23, 11, 0, 86, 88,
        3, 25, 12, 0, 87, 85, 1, 0, 0, 0, 87, 86, 1, 0, 0, 0, 87, 88, 1, 0, 0, 0, 88, 98, 1, 0, 0,
        0, 89, 91, 5, 45, 0, 0, 90, 89, 1, 0, 0, 0, 90, 91, 1, 0, 0, 0, 91, 92, 1, 0, 0, 0, 92, 98,
        3, 23, 11, 0, 93, 95, 5, 45, 0, 0, 94, 93, 1, 0, 0, 0, 94, 95, 1, 0, 0, 0, 95, 96, 1, 0, 0,
        0, 96, 98, 3, 25, 12, 0, 97, 82, 1, 0, 0, 0, 97, 90, 1, 0, 0, 0, 97, 94, 1, 0, 0, 0, 98,
        20, 1, 0, 0, 0, 99, 108, 5, 48, 0, 0, 100, 104, 7, 0, 0, 0, 101, 103, 7, 1, 0, 0, 102, 101,
        1, 0, 0, 0, 103, 106, 1, 0, 0, 0, 104, 102, 1, 0, 0, 0, 104, 105, 1, 0, 0, 0, 105, 108, 1,
        0, 0, 0, 106, 104, 1, 0, 0, 0, 107, 99, 1, 0, 0, 0, 107, 100, 1, 0, 0, 0, 108, 22, 1, 0, 0,
        0, 109, 111, 5, 46, 0, 0, 110, 112, 7, 1, 0, 0, 111, 110, 1, 0, 0, 0, 112, 113, 1, 0, 0, 0,
        113, 111, 1, 0, 0, 0, 113, 114, 1, 0, 0, 0, 114, 24, 1, 0, 0, 0, 115, 117, 7, 2, 0, 0, 116,
        118, 7, 3, 0, 0, 117, 116, 1, 0, 0, 0, 117, 118, 1, 0, 0, 0, 118, 120, 1, 0, 0, 0, 119,
        121, 7, 1, 0, 0, 120, 119, 1, 0, 0, 0, 121, 122, 1, 0, 0, 0, 122, 120, 1, 0, 0, 0, 122,
        123, 1, 0, 0, 0, 123, 26, 1, 0, 0, 0, 124, 128, 3, 13, 6, 0, 125, 127, 9, 0, 0, 0, 126,
        125, 1, 0, 0, 0, 127, 130, 1, 0, 0, 0, 128, 129, 1, 0, 0, 0, 128, 126, 1, 0, 0, 0, 129,
        131, 1, 0, 0, 0, 130, 128, 1, 0, 0, 0, 131, 132, 3, 13, 6, 0, 132, 28, 1, 0, 0, 0, 133,
        134, 5, 99, 0, 0, 134, 135, 5, 97, 0, 0, 135, 136, 5, 108, 0, 0, 136, 137, 5, 108, 0, 0,
        137, 30, 1, 0, 0, 0, 138, 142, 7, 4, 0, 0, 139, 141, 7, 5, 0, 0, 140, 139, 1, 0, 0, 0, 141,
        144, 1, 0, 0, 0, 142, 140, 1, 0, 0, 0, 142, 143, 1, 0, 0, 0, 143, 32, 1, 0, 0, 0, 144, 142,
        1, 0, 0, 0, 145, 147, 7, 6, 0, 0, 146, 145, 1, 0, 0, 0, 147, 148, 1, 0, 0, 0, 148, 146, 1,
        0, 0, 0, 148, 149, 1, 0, 0, 0, 149, 150, 1, 0, 0, 0, 150, 151, 6, 16, 0, 0, 151, 34, 1, 0,
        0, 0, 16, 0, 63, 74, 82, 87, 90, 94, 97, 104, 107, 113, 117, 122, 128, 142, 148, 1, 6, 0,
        0
    ];
}
