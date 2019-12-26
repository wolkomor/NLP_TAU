
# DO NOT USE IF YOU WANT TO LEARN SOMETHING

import re

message_printed = False
def regex_rule(regex):
    global message_printed
    if not message_printed:
        print("CATEGORIES SOLUTION USED")
        message_printed = True
    pattern = re.compile(regex)
    return lambda s: bool(pattern.match(s))

# Note - order of rules is important, first rule to match is taken
rare_words_transformation_rules = [
    ('twoDigitNum',        regex_rule('^\d{2}$')),                 # Example: 12
    ('fourDigitNum',       regex_rule('^\d{4}$')),                 # Example: 1234
    ('otherNum',           regex_rule('^\d+\.?\d+$')),             # Examples: 12345, 1234.235, 1.3
    ('commaInNum',         regex_rule('^(?:\d+,)+\d+\.?\d+$')),    # Examples: '1,234', '1,234.14'
    ('dashAndNum',         regex_rule('^[0-9]+[0-9\-]+[0-9]+$')),  # Examples: 1-1, 07-05-1998
    ('slashAndNum',        regex_rule('^[0-9]+[0-9/]+[0-9]+$')),   # Example: 10/10/2010, 1000/1000/201100
    ('ordinal',            regex_rule('^\d+(nd|rd|th)$')),         # Examples: 22nd, 53rd, 14th
    ('hour:minute',        regex_rule('^\d{1,2}:\d{2}')),          # Examples: 3:15, 12:49

    ('allCaps',            regex_rule('^[A-Z]+$')),                # Example: ALLCAP
    ('capPeriod',          regex_rule('^[A-Z]\.$')),               # Example: M.

    ('thing-and-thing',    regex_rule('^[a-z]+\-and\-[a-z]+$')),   # Example: black-and-white
    ('thing-and-thing',    regex_rule('^[a-z]+\-and\-[a-z]+$')),   # Example: black-and-white
    ('thing-than-thing',   regex_rule('^[a-z]+\-than\-[a-z]+$')),  # Example: smaller-than-expected
    ('thing-the-thing',    regex_rule('^[a-z]+\-the\-[a-z]+$')),   # Example: behind-the-scenes
    ('co-thing',               regex_rule('^co\-[a-z]+$')),        # Example: co-sponsored
    ('pre-thing',              regex_rule('^pre\-[a-z]+$')),       # Example: pre-empt
    ('pro-thing',              regex_rule('^pro\-[a-z]+$')),       # Example: pro-active
    ('much-thing',             regex_rule('^much\-[a-z]+$')),      # Example: much-publicized
    ('most-thing',             regex_rule('^most\-[a-z]+$')),      # Example: most-active
    ('low-thing',              regex_rule('^low\-[a-z]+$')),       # Example: low-level
    ('high-thing',             regex_rule('^high\-[a-z]+$')),      # Example: high-visibility
    ('inter-thing',            regex_rule('^inter\-[a-z]+$')),     # Example: inter-city
    ('-a-',                    regex_rule('^.+\-a\-.+$')),         # Example: 18-a-share

    ('iedLowercase',           regex_rule('^[a-z]+ied$')),         # Example: supplied
    ('edLowercase',            regex_rule('^[a-z]+ed$')),          # Example: played
    ('ingLowercase',           regex_rule('^[a-z]+ing$')),         # Example: playing
    ('tionLowercase',          regex_rule('^[a-z]+tion$')),        # Example: transition
    ('sionLowercase',          regex_rule('^[a-z]+sion$')),        # Example: emission
    ('xionLowercase',          regex_rule('^[a-z]+xion$')),        # Example: complexion
    ('ableLowercase',          regex_rule('^[a-z]+able$')),        # Example: formidable
    ('ibleLowercase',          regex_rule('^[a-z]+ible$')),        # Example: tangible
    ('fulLowercase',           regex_rule('^[a-z]+ful$')),         # Example: powerful
    ('anceLowercase',          regex_rule('^[a-z]+ance$')),        # Example: performance
    ('enceLowercase',          regex_rule('^[a-z]+ence$')),        # Example: intelligence
    ('sialLowercase',          regex_rule('^[a-z]+sial$')),        # Example: controversial
    ('tialLowercase',          regex_rule('^[a-z]+tial$')),        # Example: potential
    ('mentLowercase',          regex_rule('^[a-z]+ment$')),        # Example: establishment
    ('shipLowercase',          regex_rule('^[a-z]+ship$')),        # Example: relationship
    ('nessLowercase',          regex_rule('^[a-z]+ness$')),        # Example: kindness
    ('hoodLowercase',          regex_rule('^[a-z]+hood$')),        # Example: neighborhood
    ('domLowercase',           regex_rule('^[a-z]+dom$')),         # Example: kingdom
    ('eeLowercase',            regex_rule('^[a-z]+ee$')),          # Example: trainee
    ('istLowercase',           regex_rule('^[a-z]+ist$')),         # Example: socialist
    ('ismLowercase',           regex_rule('^[a-z]+ism$')),         # Example: capitalism
    ('ageLowercase',           regex_rule('^[a-z]+age$')),         # Example: village
    ('erLowercase',            regex_rule('^[a-z]+er$')),          # Example: driver
    ('orLowercase',            regex_rule('^[a-z]+or$')),          # Example: director
    ('ityLowercase',           regex_rule('^[a-z]+ity$')),         # Example: equality
    ('tyLowercase',            regex_rule('^[a-z]+ty$')),          # Example: cruelty
    ('ryLowercase',            regex_rule('^[a-z]+ry$')),          # Example: robbery
    ('lyLowercase',            regex_rule('^[a-z]+ly$')),          # Example: easily
    ('wardLowercase',          regex_rule('^[a-z]+ward$')),        # Example: backward
    ('wardsLowercase',         regex_rule('^[a-z]+wards$')),       # Example: backwards
    ('izeLowercase',           regex_rule('^[a-z]+ize$')),         # Example: characterize
    ('iseLowercase',           regex_rule('^[a-z]+ise$')),         # Example: characterise (UK)
    ('ifyLowercase',           regex_rule('^[a-z]+ify$')),         # Example: signify
    ('ateLowercase',           regex_rule('^[a-z]+ate$')),         # Example: irrigate
    ('enLowercase',            regex_rule('^[a-z]+en$')),          # Example: strengthen
    ('icLowercase',            regex_rule('^[a-z]+ic$')),          # Example: classic
    ('alLowercase',            regex_rule('^[a-z]+al$')),          # Example: brutal
    ('yLowercase',             regex_rule('^[a-z]+y$')),           # Example: cloudy
    ('estLowercase',           regex_rule('^[a-z]+est$')),         # Example: strongest
    ('ianLowercase',           regex_rule('^[a-z]+ian$')),         # Example: utilitarian
    ('iveLowercase',           regex_rule('^[a-z]+ive$')),         # Example: productive
    ('ishLowercase',           regex_rule('^[a-z]+ish$')),         # Example: childish
    ('lessLowercase',          regex_rule('^[a-z]+less$')),        # Example: useless
    ('ousLowercase',           regex_rule('^[a-z]+ous$')),         # Example: nervous

    ('otherLowercase',         regex_rule('^[a-z]+$')),            # Example: abc

    ('eseNationality',         regex_rule('^[A-Z][a-z]+ese$')),    # Example: Japanese
    ('ishNationality',         regex_rule('^[A-Z][a-z]+ish$')),    # Example: Spanish
    ('ianNationality',         regex_rule('^[A-Z][a-z]+ian$')),    # Example: Canadian

    ('initCap_iedLowercase',   regex_rule('^[A-Z][a-z]+ied$')),    # Example: Supplied
    ('initCap_edLowercase',    regex_rule('^[A-Z][a-z]+ed$')),     # Example: Played
    ('initCap_ingLowercase',   regex_rule('^[A-Z][a-z]+ing$')),    # Example: Playing
    ('initCap_tionLowercase',  regex_rule('^[A-Z][a-z]+tion$')),   # Example: Transition
    ('initCap_sionLowercase',  regex_rule('^[A-Z][a-z]+sion$')),   # Example: Emission
    ('initCap_xionLowercase',  regex_rule('^[A-Z][a-z]+xion$')),   # Example: Complexion
    ('initCap_ableLowercase',  regex_rule('^[A-Z][a-z]+able$')),   # Example: Formidable
    ('initCap_ibleLowercase',  regex_rule('^[A-Z][a-z]+ible$')),   # Example: Tangible
    ('initCap_fulLowercase',   regex_rule('^[A-Z][a-z]+ful$')),    # Example: Powerful
    ('initCap_anceLowercase',  regex_rule('^[A-Z][a-z]+ance$')),   # Example: Performance
    ('initCap_enceLowercase',  regex_rule('^[A-Z][a-z]+ence$')),   # Example: Intelligence
    ('initCap_sialLowercase',  regex_rule('^[A-Z][a-z]+sial$')),   # Example: Controversial
    ('initCap_tialLowercase',  regex_rule('^[A-Z][a-z]+tial$')),   # Example: Potential
    ('initCap_mentLowercase',  regex_rule('^[A-Z][a-z]+ment$')),   # Example: Establishment
    ('initCap_shipLowercase',  regex_rule('^[A-Z][a-z]+ship$')),   # Example: Relationship
    ('initCap_nessLowercase',  regex_rule('^[A-Z][a-z]+ness$')),   # Example: Kindness
    ('initCap_hoodLowercase',  regex_rule('^[A-Z][a-z]+hood$')),   # Example: Neighborhood
    ('initCap_domLowercase',   regex_rule('^[A-Z][a-z]+dom$')),    # Example: Kingdom
    ('initCap_eeLowercase',    regex_rule('^[A-Z][a-z]+ee$')),     # Example: Trainee
    ('initCap_istLowercase',   regex_rule('^[A-Z][a-z]+ist$')),    # Example: Socialist
    ('initCap_ismLowercase',   regex_rule('^[A-Z][a-z]+ism$')),    # Example: Capitalism
    ('initCap_ageLowercase',   regex_rule('^[A-Z][a-z]+age$')),    # Example: Village
    ('initCap_erLowercase',    regex_rule('^[A-Z][a-z]+er$')),     # Example: Driver
    ('initCap_orLowercase',    regex_rule('^[A-Z][a-z]+or$')),     # Example: Director
    ('initCap_ityLowercase',   regex_rule('^[A-Z][a-z]+ity$')),    # Example: Equality
    ('initCap_tyLowercase',    regex_rule('^[A-Z][a-z]+ty$')),     # Example: Cruelty
    ('initCap_ryLowercase',    regex_rule('^[A-Z][a-z]+ry$')),     # Example: Robbery
    ('initCap_lyLowercase',    regex_rule('^[A-Z][a-z]+ly$')),     # Example: Easily
    ('initCap_wardLowercase',  regex_rule('^[A-Z][a-z]+ward$')),   # Example: Backward
    ('initCap_wardsLowercase', regex_rule('^[A-Z][a-z]+wards$')),  # Example: Backwards
    ('initCap_izeLowercase',   regex_rule('^[A-Z][a-z]+ize$')),    # Example: Characterize
    ('initCap_iseLowercase',   regex_rule('^[A-Z][a-z]+ise$')),    # Example: Characterise (UK)
    ('initCap_ifyLowercase',   regex_rule('^[A-Z][a-z]+ify$')),    # Example: Signify
    ('initCap_ateLowercase',   regex_rule('^[A-Z][a-z]+ate$')),    # Example: Irrigate
    ('initCap_enLowercase',    regex_rule('^[A-Z][a-z]+en$')),     # Example: Strengthen
    ('initCap_icLowercase',    regex_rule('^[A-Z][a-z]+ic$')),     # Example: Classic
    ('initCap_alLowercase',    regex_rule('^[A-Z][a-z]+al$')),     # Example: Brutal
    ('initCap_yLowercase',     regex_rule('^[A-Z][a-z]+y$')),      # Example: Cloudy

    ('initCap', regex_rule('^[A-Z].*$'))  # Example: Cap
]
