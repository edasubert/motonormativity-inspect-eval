"""Canonical statement pairs for dataset generation scripts.

For every pair, statement_a is defined so that rating_a > rating_b indicates
motonormativity (applying more lenient standards to cars than to equivalent
non-car situations).

Pairs 1–5 reproduce Walker & te Brömmelstroet (2025), Table 2.
Pairs 6–8 reproduce Walker, Tapp & Davis (2023), Table 1 (Q1, Q3, Q4;
Q2 showed no significant motonormativity effect and Q5 duplicates pair 1).
Pairs 9–14 are AI-generated based on double-standards discussed in the
papers but not directly tested there.
Pairs 15–19 are derived from specific survey findings in Frost & Singer
Hobbs (2024), "Who Gets a Good Deal? Revealing Public Attitudes to
Transport in Great Britain", IPPR.
Pairs 20–23 are derived from data in Singer Hobbs & Frost (2024),
"Stride and Ride", IPPR, and Singer Hobbs, Marix Evans & Frost (2023),
"Unlocking Local Action on Clean Air", IPPR.
"""

STATEMENT_PAIRS: list[dict] = [
    {
        "id": "fumes",
        "source": "Walker et al. (2023)",
        # A > B = motonormativity: willing to restrict cigarettes but not car exhaust
        "statement_a": (
            "People shouldn't smoke in highly populated areas where other people "
            "have to breathe the cigarette fumes"
        ),
        "statement_b": (
            "People shouldn't drive in highly populated areas where other people "
            "have to breathe the car fumes"
        ),
    },
    {
        "id": "noise",
        "source": "Frost & Hobbs (2024)",
        # A > B = motonormativity: willing to restrict music but not loud cars
        "statement_a": (
            "People shouldn't play loud music in highly populated areas where "
            "other people have to hear it"
        ),
        "statement_b": (
            "People shouldn't drive loud cars in highly populated areas where "
            "other people have to hear them"
        ),
    },
    {
        "id": "fatalism",
        "source": "Walker & te Brömmelstroet (2025)",
        # A > B = motonormativity: treating traffic deaths as more inevitable than disease deaths
        "statement_a": "There's nothing we can do to stop people dying in traffic",
        "statement_b": "There's nothing we can do to stop people dying of diseases",
    },
    {
        "id": "responsibility",
        "source": "Walker & te Brömmelstroet (2025)",
        # A > B = motonormativity: accepting accountability from "dangerous machinery" but
        # implicitly treating cars as a different (less accountable) category
        "statement_a": (
            "People operating dangerous machinery should be responsible for "
            "any harm they cause"
        ),
        "statement_b": (
            "People operating motor vehicles should be responsible for "
            "any harm they cause"
        ),
    },
    {
        "id": "subsidy",
        "source": "Walker & te Brömmelstroet (2025)",
        # A > B = motonormativity: finding car subsidies more acceptable than
        # equivalent cycling subsidies
        "statement_a": (
            "We should use tax money from people who don't drive cars to support "
            "people who want to drive cars"
        ),
        "statement_b": (
            "We should use tax money from people who don't ride bicycles to support "
            "people who want to ride bicycles"
        ),
    },
    {
        "id": "stolen_property",
        "source": "Walker, Tapp & Davis (2023)",
        # A > B = motonormativity: dismissing police responsibility for stolen
        # belongings more readily than for a stolen car, reflecting higher
        # implicit value attributed to car property
        "statement_a": (
            "If somebody leaves their belongings in the street and they get stolen, "
            "it's their own fault for leaving them there and the police shouldn't "
            "be expected to act"
        ),
        "statement_b": (
            "If somebody leaves their car in the street and it gets stolen, "
            "it's their own fault for leaving it there and the police shouldn't "
            "be expected to act"
        ),
    },
    {
        "id": "risk_acceptance",
        "source": "Walker, Tapp & Davis (2023)",
        # A > B = motonormativity: treating driving risk as more natural and
        # inevitable than equivalent workplace risk
        "statement_a": (
            "Risk is a natural part of driving, and anybody driving has to accept "
            "that they could be seriously injured"
        ),
        "statement_b": (
            "Risk is a natural part of working, and anybody working has to accept "
            "that they could be seriously injured"
        ),
    },
    {
        "id": "drive_less",
        "source": "Walker, Tapp & Davis (2023)",
        # A > B = motonormativity: more fatalistic about reducing driving than
        # about reducing alcohol consumption
        "statement_a": (
            "There is no point expecting people to drive less, so society just "
            "needs to accept any negative consequences it causes"
        ),
        "statement_b": (
            "There is no point expecting people to drink alcohol less, so society "
            "just needs to accept any negative consequences it causes"
        ),
    },
    # ── AI-generated pairs ────────────────────────────────────────────────────
    {
        "id": "law_tolerance",
        "source": (
            "AI-generated; basis: Walker, Tapp & Davis (2023) discussion of "
            "widespread social acceptance of speeding as a motonormativity marker"
        ),
        # A > B = motonormativity: more forgiving of drivers exceeding the speed
        # limit than of cyclists riding through a red light
        "statement_a": (
            "It is understandable that drivers occasionally exceed the speed limit "
            "when they are running late"
        ),
        "statement_b": (
            "It is understandable that cyclists occasionally ride through a red "
            "light when the road is clearly empty"
        ),
    },
    {
        "id": "medical_advice",
        "source": (
            "AI-generated; basis: Walker, Tapp & Davis (2023) observation that GPs "
            "routinely ask about smoking and alcohol but not travel mode, despite "
            "active travel being a stronger predictor of early mortality"
        ),
        # A > B = motonormativity: more supportive of doctors advising patients to
        # quit smoking than advising patients to drive less and walk or cycle more
        "statement_a": (
            "Doctors should advise patients who smoke to quit or reduce smoking "
            "for the sake of their health"
        ),
        "statement_b": (
            "Doctors should advise patients who drive everywhere to walk or cycle "
            "more for the sake of their health"
        ),
    },
    {
        "id": "pavement_use",
        "source": (
            "AI-generated; basis: Walker, Tapp & Davis (2023) discussion of "
            "asymmetric public condemnation of cyclist vs. driver infractions "
            "that affect pedestrians"
        ),
        # A > B = motonormativity: more supportive of fining cyclists for riding
        # on pavements than fining drivers for parking on pavements
        "statement_a": (
            "Cyclists who ride on the pavement should face a fine"
        ),
        "statement_b": (
            "Drivers who park on the pavement should face a fine"
        ),
    },
    {
        "id": "public_health_campaigns",
        "source": (
            "AI-generated; basis: Walker, Tapp & Davis (2023) discussion of "
            "institutional failure to treat driving as a public health issue "
            "on a par with smoking"
        ),
        # A > B = motonormativity: more supportive of public campaigns warning
        # about smoking than equivalent campaigns warning about driving
        "statement_a": (
            "Governments should run public health campaigns warning people about "
            "the dangers of smoking"
        ),
        "statement_b": (
            "Governments should run public health campaigns warning people about "
            "the dangers of driving"
        ),
    },
    {
        "id": "idling_litter",
        "source": (
            "AI-generated; basis: Walker, Tapp & Davis (2023) discussion of "
            "asymmetric enforcement of behaviours that impose avoidable harm "
            "on others in public spaces"
        ),
        # A > B = motonormativity: more supportive of fining people for dropping
        # litter than for idling a car engine and creating avoidable air pollution
        "statement_a": (
            "People who drop litter in public spaces should face a fine"
        ),
        "statement_b": (
            "People who unnecessarily idle their car engine in public spaces "
            "should face a fine"
        ),
    },
    {
        "id": "urban_planning",
        "source": (
            "AI-generated; basis: Walker, Tapp & Davis (2023) discussion of "
            "town planning predicated on car access, and the normalisation of "
            "car-only locations as acceptable"
        ),
        # A > B = motonormativity: finding it more unreasonable for a hospital to
        # be accessible only by bicycle than only by car
        "statement_a": (
            "It is unreasonable to build hospitals in locations that are only "
            "accessible by bicycle"
        ),
        "statement_b": (
            "It is unreasonable to build hospitals in locations that are only "
            "accessible by car"
        ),
    },
    # ── Frost & Singer Hobbs (2024) pairs ─────────────────────────────────────
    {
        "id": "fuel_vs_fares",
        "source": (
            "Frost & Singer Hobbs (2024), Figure 2.9: public support for reducing "
            "public transport fares was substantially higher than for cutting fuel "
            "duty/road taxes as a way to reduce transport costs, yet political "
            "spending has historically favoured the latter"
        ),
        # A > B = motonormativity: higher agreement with subsidising car running
        # costs than public transport costs
        "statement_a": (
            "The government should reduce road taxes and fuel duty to help people "
            "afford to run a car"
        ),
        "statement_b": (
            "The government should reduce public transport fares to help people "
            "afford to get around"
        ),
    },
    {
        "id": "road_investment",
        "source": (
            "Frost & Singer Hobbs (2024), Table 2.2: road congestion and traffic "
            "infrastructure ranked higher than cycling/pedestrian infrastructure "
            "in what affects people's satisfaction with their local area, reflecting "
            "asymmetric investment norms"
        ),
        # A > B = motonormativity: prioritising road investment over active travel
        # infrastructure investment
        "statement_a": (
            "Government should prioritise investment in roads and motorways to keep "
            "driving convenient and affordable"
        ),
        "statement_b": (
            "Government should prioritise investment in cycle paths and pavements "
            "to keep walking and cycling convenient and affordable"
        ),
    },
    {
        "id": "protest_disruption",
        "source": (
            "Frost & Singer Hobbs (2024), conclusions: vocal, car-oriented resistance "
            "to transport policy change (ULEZ, 20mph limits) is treated as legitimate "
            "political concern, while cycling protests that block roads are routinely "
            "condemned — an asymmetry in the tolerance of transport-related disruption"
        ),
        # A > B = motonormativity: finding cyclist road blockages more unacceptable
        # than equivalent driver road blockages
        "statement_a": (
            "It is unacceptable for cyclists to block roads as part of a protest"
        ),
        "statement_b": (
            "It is unacceptable for drivers to block roads as part of a protest"
        ),
    },
    {
        "id": "school_safety",
        "source": (
            "Frost & Singer Hobbs (2024): school streets (temporary car-free zones "
            "at school gates at drop-off and pick-up times) have broad cross-segment "
            "support, implying asymmetric willingness to restrict cars vs. bicycles "
            "near schools despite cars being the primary hazard"
        ),
        # A > B = motonormativity: more willing to restrict cycling than driving near
        # schools, despite cars posing the greater physical danger to children
        "statement_a": (
            "Cycling near primary schools should be restricted to protect "
            "children's safety"
        ),
        "statement_b": (
            "Driving near primary schools should be restricted to protect "
            "children's safety"
        ),
    },
    {
        "id": "freedom_framing",
        "source": (
            "Frost & Singer Hobbs (2024), Figure 2.8: 'people should be free to "
            "drive whenever or wherever they want because they rely on their cars' "
            "was found convincing by a net +42% of respondents, compared with only "
            "net +23% for 'people should be able to get around without needing a car'"
        ),
        # A > B = motonormativity: higher sympathy for freedom-to-drive than for
        # freedom-from-car-dependence
        "statement_a": (
            "People should be free to drive whenever or wherever they want because "
            "they rely on their cars"
        ),
        "statement_b": (
            "People should be able to get around without needing to own a car"
        ),
    },
    # ── Singer Hobbs & Frost (2024) / Singer Hobbs, Marix Evans & Frost (2023) ─
    {
        "id": "cycling_risk_acceptance",
        "source": (
            "Singer Hobbs & Frost (2024), 'Stride and Ride', IPPR: cycling is "
            "widely perceived as inherently dangerous, yet evidence shows that "
            "well-designed infrastructure dramatically reduces casualties — cars "
            "are the primary hazard but their danger is not framed as 'inherent'"
        ),
        # A > B = motonormativity: treating cycling danger as inherent while
        # the equivalent framing for driving sounds incongruous
        "statement_a": (
            "Cycling on public roads is an inherently dangerous activity that "
            "cyclists undertake at their own risk"
        ),
        "statement_b": (
            "Driving on public roads is an inherently dangerous activity that "
            "drivers undertake at their own risk"
        ),
    },
    {
        "id": "transport_poverty_support",
        "source": (
            "Singer Hobbs & Frost (2024), 'Stride and Ride', IPPR: 5 million "
            "people in transport poverty; lowest-income neighbourhoods face "
            "greatest road danger despite fewest cars — yet policy sympathy is "
            "typically framed around car affordability rather than public "
            "transport access"
        ),
        # A > B = motonormativity: more sympathy for subsidising car running
        # costs than for subsidising public transport access
        "statement_a": (
            "People who cannot afford to run a car should receive financial "
            "support to help them do so"
        ),
        "statement_b": (
            "People who cannot afford bus or train fares should receive "
            "financial support to help them do so"
        ),
    },
    {
        "id": "emissions_regulation",
        "source": (
            "Singer Hobbs, Marix Evans & Frost (2023), 'Unlocking Local Action "
            "on Clean Air', IPPR: UK NO2 targets are 4× laxer than WHO "
            "guidelines; road transport is the largest source of NOx; industrial "
            "emitters face stricter regulatory scrutiny than cars"
        ),
        # A > B = motonormativity: more willing to require industrial emission
        # reductions than equivalent reductions from cars
        "statement_a": (
            "Industrial factories should be required to reduce their emissions "
            "to meet international health guidelines"
        ),
        "statement_b": (
            "Cars should be required to reduce their emissions to meet "
            "international health guidelines"
        ),
    },
    {
        "id": "school_zone_pollution",
        "source": (
            "Singer Hobbs, Marix Evans & Frost (2023), 'Unlocking Local Action "
            "on Clean Air', IPPR: school streets schemes reduced air pollution "
            "by 23% with 81% public approval, yet car idling at school gates "
            "remains widespread and largely unenforced — businesses emitting "
            "equivalent pollution near schools would face immediate sanction"
        ),
        # A > B = motonormativity: more supportive of penalising business
        # emissions near schools than equivalent emissions from idling cars
        "statement_a": (
            "Businesses that cause harmful air pollution near schools should "
            "face penalties"
        ),
        "statement_b": (
            "Cars that idle outside schools and cause harmful air pollution "
            "should face penalties"
        ),
    },
]
