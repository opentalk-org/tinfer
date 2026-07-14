use super::Case;

pub(super) const CASES: &[Case] = &[
    Case {
        text: "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'\nSo she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
        expected: &[
            "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'",
            "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
        ],
    },
    Case {
        text: "'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)",
        expected: &[
            "'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)",
        ],
    },
    Case {
        text: "Down, down, down. Would the fall NEVER come to an ! 'I wonder how many miles I've fallen by this time?' she said aloud.",
        expected: &[
            "Down, down, down.",
            "Would the fall NEVER come to an !",
            "'I wonder how many miles I've fallen by this time?' she said aloud.",
        ],
    },
    Case {
        text: "Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled 'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody, so managed to put it into one of the cupboards as she fell past it. 'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)",
        expected: &[
            "Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next.",
            "First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs.",
            "She took down a jar from one of the shelves as she passed; it was labelled 'ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody, so managed to put it into one of the cupboards as she fell past it.",
            "'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)",
        ],
    },
    Case {
        text: "A minute is a unit of measurement of time or of angle. The minute is a unit of time equal to 1/60th of an hour or 60 seconds by 1. In the UTC time scale, a minute occasionally has 59 or 61 seconds; see leap second. The minute is not an SI unit; however, it is accepted for use with SI units. The symbol for minute or minutes is min. The fact that an hour contains 60 minutes is probably due to influences from the Babylonians, who used a base-60 or sexagesimal counting system. Colloquially, a min. may also refer to an indefinite amount of time substantially longer than the standardized length.",
        expected: &[
            "A minute is a unit of measurement of time or of angle.",
            "The minute is a unit of time equal to 1/60th of an hour or 60 seconds by 1.",
            "In the UTC time scale, a minute occasionally has 59 or 61 seconds; see leap second.",
            "The minute is not an SI unit; however, it is accepted for use with SI units.",
            "The symbol for minute or minutes is min.",
            "The fact that an hour contains 60 minutes is probably due to influences from the Babylonians, who used a base-60 or sexagesimal counting system.",
            "Colloquially, a min. may also refer to an indefinite amount of time substantially longer than the standardized length.",
        ],
    },
    Case { text: "It was a cold \nnight in the city.", expected: &["It was a cold night in the city."] },
    Case { text: "features\ncontact manager\nevents, activities\n", expected: &["features", "contact manager", "events, activities"] },
    Case {
        text: "Hello world.Today is Tuesday.Mr. Smith went to the store and bought 1,000.That is a lot.",
        expected: &["Hello world.", "Today is Tuesday.", "Mr. Smith went to the store and bought 1,000.", "That is a lot."],
    },
    Case {
        text: "About Me...............................................................................................5\n        Chapter 2 ...................................................................... 6\n        Three Weeks Later............................................................................ 7\n        Better Eating........................................................................................ 8\n        What's the Score?.............................................................. 9\n        How To Calculate the Score................... 16-17",
        expected: &["About Me", "Chapter 2", "Three Weeks Later", "Better Eating", "What's the Score?", "How To Calculate the Score"],
    },
    Case { text: "I think Jun. is a great month, said Mr. Suzuki.", expected: &["I think Jun. is a great month, said Mr. Suzuki."] },
    Case { text: "Jun. is a great month, said Mr. Suzuki.", expected: &["Jun. is a great month, said Mr. Suzuki."] },
    Case { text: "I have 1.000.00. Yay $.50 and .50! That's 600.", expected: &["I have 1.000.00.", "Yay $.50 and .50!", "That's 600."] },
    Case { text: "1.) This is a list item with a parens.", expected: &["1.) This is a list item with a parens."] },
    Case { text: "1. This is a list item.", expected: &["1. This is a list item."] },
    Case { text: "I live in the U.S.A. I went to J.C. Penney.", expected: &["I live in the U.S.A.", "I went to J.C. Penney."] },
    Case { text: "His name is Alfred E. Sloan.", expected: &["His name is Alfred E. Sloan."] },
    Case {
        text: "Q. What is his name? A. His name is Alfred E. Sloan.",
        expected: &["Q. What is his name?", "A. His name is Alfred E. Sloan."],
    },
    Case { text: "Today is 11.18.2014.", expected: &["Today is 11.18.2014."] },
    Case {
        text: "I need you to find 3 items, e.g. a hat, a coat, and a bag.",
        expected: &["I need you to find 3 items, e.g. a hat, a coat, and a bag."],
    },
    Case {
        text: "The game is the Giants vs. the Tigers at 10 p.m. I'm going are you?",
        expected: &["The game is the Giants vs. the Tigers at 10 p.m.", "I'm going are you?"],
    },
    Case { text: "He is no. 5, the shortstop.", expected: &["He is no. 5, the shortstop."] },
    Case { text: "Remove long strings of dots........please.", expected: &["Remove long strings of dots please."] },
    Case {
        text: "See our additional services section or contact us for pricing\n.\n\n\nPricing Additionl Info\n",
        expected: &["See our additional services section or contact us for pricing.", "Pricing Additionl Info"],
    },
    Case {
        text: "As payment for 1. above, pay us a commission fee of 0 yen and for 2. above, no fee will be paid.",
        expected: &["As payment for 1. above, pay us a commission fee of 0 yen and for 2. above, no fee will be paid."],
    },
    Case {
        text: "See our additional services section or contact us for pricing\n. Pricing Additionl Info",
        expected: &["See our additional services section or contact us for pricing.", "Pricing Additionl Info"],
    },
    Case { text: "I have 600. How many do you have?", expected: &["I have 600.", "How many do you have?"] },
    Case { text: "\n3\n\nIntroduction\n\n", expected: &["3", "Introduction"] },
    Case { text: "\nW\nA\nRN\nI\nNG\n", expected: &["WARNING"] },
    Case { text: "\n\n\nW\nA\nRN\nI\nNG\n \n/\n \nA\nV\nE\nR\nT\nI\nS\nE\nM\nE\nNT\n", expected: &["WARNING", "/", "AVERTISEMENT"] },
    Case {
        text: "\"Help yourself, sweetie,\" shouted Candy and gave her the cookie.",
        expected: &["\"Help yourself, sweetie,\" shouted Candy and gave her the cookie."],
    },
    Case {
        text: "Until its release, a generic mechanism was known, where the sear keeps the hammer in back position, and when one pulls the trigger, the sear slips out of hammer’s notches, the hammer falls initiating \na shot.",
        expected: &[
            "Until its release, a generic mechanism was known, where the sear keeps the hammer in back position, and when one pulls the trigger, the sear slips out of hammer’s notches, the hammer falls initiating a shot.",
        ],
    },
    Case {
        text: "This is a test. Until its release, a generic mechanism was known, where the sear keeps the hammer in back position, and when one pulls the trigger, the sear slips out of hammer’s notches, the hammer falls initiating \na shot.",
        expected: &[
            "This is a test.",
            "Until its release, a generic mechanism was known, where the sear keeps the hammer in back position, and when one pulls the trigger, the sear slips out of hammer’s notches, the hammer falls initiating a shot.",
        ],
    },
    Case {
        text: "This was because it was an offensive weapon, designed to fight at a distance up to 400 yd \n( 365.8 m ).",
        expected: &["This was because it was an offensive weapon, designed to fight at a distance up to 400 yd ( 365.8 m )."],
    },
    Case {
        text: "“Are demonstrations are evidence of the public anger and frustration at opaque environmental management and decision-making?” Others yet say: \"Should we be scared about these 'protests'?\"",
        expected: &[
            "“Are demonstrations are evidence of the public anger and frustration at opaque environmental management and decision-making?”",
            "Others yet say: \"Should we be scared about these 'protests'?\"",
        ],
    },
    Case { text: "www.testurl.Awesome.com", expected: &["www.testurl.Awesome.com"] },
    Case { text: "http://testurl.Awesome.com", expected: &["http://testurl.Awesome.com"] },
    Case { text: "St. Michael's Church in is a church.", expected: &["St. Michael's Church in is a church."] },
    Case { text: "JFK Jr.'s book is on sale.", expected: &["JFK Jr.'s book is on sale."] },
    Case {
        text: "This is e.g. Mr. Smith, who talks slowly... And this is another sentence.",
        expected: &["This is e.g. Mr. Smith, who talks slowly...", "And this is another sentence."],
    },
    Case {
        text: "Leave me alone!, he yelled. I am in the U.S. Army. Charles (Ind.) said he.",
        expected: &["Leave me alone!, he yelled.", "I am in the U.S. Army.", "Charles (Ind.) said he."],
    },
    Case {
        text: "This is the U.S. Senate my friends. <em>Yes.</em> <em>It is</em>!",
        expected: &["This is the U.S. Senate my friends.", "Yes.", "It is!"],
    },
    Case { text: "Send it to P.O. box 6554", expected: &["Send it to P.O. box 6554"] },
    Case {
        text: "There were 500 cases in the U.S. The U.S. Commission asked the U.S. Government to give their opinion on the issue.",
        expected: &[
            "There were 500 cases in the U.S.",
            "The U.S. Commission asked the U.S. Government to give their opinion on the issue.",
        ],
    },
    Case {
        text: "CELLULAR COMMUNICATIONS INC. sold 1,550,000 common shares at $21.75 each yesterday, according to lead underwriter L.F. Rothschild & Co. (cited from WSJ 05/29/1987)",
        expected: &[
            "CELLULAR COMMUNICATIONS INC. sold 1,550,000 common shares at $21.75 each yesterday, according to lead underwriter L.F. Rothschild & Co. (cited from WSJ 05/29/1987)",
        ],
    },
    Case {
        text: "Rolls-Royce Motor Cars Inc. said it expects its U.S. sales to remain steady at about 1,200 cars in 1990. `So what if you miss 50 tanks somewhere?' asks Rep. Norman Dicks (D., Wash.), a member of the House group that visited the talks in Vienna. Later, he recalls the words of his Marxist mentor: `The people! Theft! The holy fire!'",
        expected: &[
            "Rolls-Royce Motor Cars Inc. said it expects its U.S. sales to remain steady at about 1,200 cars in 1990.",
            "'So what if you miss 50 tanks somewhere?' asks Rep. Norman Dicks (D., Wash.), a member of the House group that visited the talks in Vienna.",
            "Later, he recalls the words of his Marxist mentor: 'The people! Theft! The holy fire!'",
        ],
    },
    Case { text: "He climbed Mt. Fuji.", expected: &["He climbed Mt. Fuji."] },
    Case {
        text: "He speaks !Xũ, !Kung, ǃʼOǃKung, !Xuun, !Kung-Ekoka, ǃHu, ǃKhung, ǃKu, ǃung, ǃXo, ǃXû, ǃXung, ǃXũ, and !Xun.",
        expected: &["He speaks !Xũ, !Kung, ǃʼOǃKung, !Xuun, !Kung-Ekoka, ǃHu, ǃKhung, ǃKu, ǃung, ǃXo, ǃXû, ǃXung, ǃXũ, and !Xun."],
    },
    Case {
        text: "Test strange period．Does it segment correctly．", expected: &["Test strange period．", "Does it segment correctly．"]
    },
    Case {
        text: "<h2 class=\"lined\">Hello</h2>\n<p>This is a test. Another test.</p>\n<div class=\"center\"><p>\n<img src=\"/images/content/example.jpg\">\n</p></div>",
        expected: &["Hello", "This is a test.", "Another test."],
    },
    Case {
        text: "This sentence ends with the psuedo-number x10. This one with the psuedo-number %3.00. One last sentence.",
        expected: &["This sentence ends with the psuedo-number x10.", "This one with the psuedo-number %3.00.", "One last sentence."],
    },
    Case {
        text: "Testing mixed numbers Jahr10. And another 0.3 %11. That's weird.",
        expected: &["Testing mixed numbers Jahr10.", "And another 0.3 %11.", "That's weird."],
    },
    Case { text: "Were Jane and co. at the party?", expected: &["Were Jane and co. at the party?"] },
    Case { text: "St. Michael's Church is on 5th st. near the light.", expected: &["St. Michael's Church is on 5th st. near the light."] },
    Case { text: "Let's ask Jane and co. They should know.", expected: &["Let's ask Jane and co.", "They should know."] },
    Case { text: "He works at Yahoo! and Y!J.", expected: &["He works at Yahoo! and Y!J."] },
    Case { text: "The Scavenger Hunt ends on Dec. 31st, 2011.", expected: &["The Scavenger Hunt ends on Dec. 31st, 2011."] },
    Case {
        text: "Putter King Scavenger Hunt Trophy\n(6 3/4\" Engraved Crystal Trophy - Picture Coming Soon)\nThe Putter King team will judge the scavenger hunt and all decisions will be final.  The scavenger hunt is open to anyone and everyone.  The scavenger hunt ends on Dec. 31st, 2011.",
        expected: &[
            "Putter King Scavenger Hunt Trophy",
            "(6 3/4\" Engraved Crystal Trophy - Picture Coming Soon)",
            "The Putter King team will judge the scavenger hunt and all decisions will be final.",
            "The scavenger hunt is open to anyone and everyone.",
            "The scavenger hunt ends on Dec. 31st, 2011.",
        ],
    },
    Case {
        text: "Unauthorized modifications, alterations or installations of or to this equipment are prohibited and are in violation of AR 750-10. Any such unauthorized modifications, alterations or installations could result in death, injury or damage to the equipment.",
        expected: &[
            "Unauthorized modifications, alterations or installations of or to this equipment are prohibited and are in violation of AR 750-10.",
            "Any such unauthorized modifications, alterations or installations could result in death, injury or damage to the equipment.",
        ],
    },
    Case {
        text: "Header 1.2; Attachment Z\n\n\td. Compliance Log – Volume 12 \n\tAttachment A\n\n\te. Additional Logistics Data\n\tSection 10",
        expected: &[
            "Header 1.2; Attachment Z",
            "d. Compliance Log – Volume 12",
            "Attachment A",
            "e. Additional Logistics Data",
            "Section 10",
        ],
    },
    Case {
        text: "a.) The first item b.) The second item c.) The third list item",
        expected: &["a.) The first item", "b.) The second item", "c.) The third list item"],
    },
    Case {
        text: "a) The first item b) The second item c) The third list item",
        expected: &["a) The first item", "b) The second item", "c) The third list item"],
    },
    Case {
        text: "Hello Wolrd. Here is a secret code AS750-10. Another sentence. Finally, this. 1. The first item 2. The second item 3. The third list item 4. Hello 5. Hello 6. Hello 7. Hello 8. Hello 9. Hello 10. Hello 11. Hello",
        expected: &[
            "Hello Wolrd.",
            "Here is a secret code AS750-10.",
            "Another sentence.",
            "Finally, this.",
            "1. The first item",
            "2. The second item",
            "3. The third list item",
            "4. Hello",
            "5. Hello",
            "6. Hello",
            "7. Hello",
            "8. Hello",
            "9. Hello",
            "10. Hello",
            "11. Hello",
        ],
    },
];
