"""
Generates a labeled sample dataset of real-looking and fake-looking news
articles for training/demoing the detector.

NOTE: This is a *synthetic demo dataset* built from templates so the whole
pipeline can be trained and tested without needing to download an external
corpus. For production use, swap this out for a real corpus such as:
  - LIAR dataset (politifact statements)
  - ISOT Fake News Dataset
  - FakeNewsNet
  - Kaggle "Fake and real news dataset"
The loader in src/data_utils.py can load any CSV with `text,label,source` columns,
so plugging in a real dataset is a one-line change.
"""
import csv
import random

random.seed(42)

REAL_TEMPLATES = [
    "The {org} reported on {day} that {metric} rose by {pct}% in the {period}, "
    "according to data released this week. Analysts at {org2} said the figures "
    "were broadly in line with expectations, though they cautioned that "
    "{caveat}. The report is based on {n} data points collected across "
    "{region} between {month1} and {month2}.",

    "Officials from the {org} confirmed on {day} that {policy} will take effect "
    "starting {month1}. In a statement, a spokesperson said the change follows "
    "a review that began in {month2} and involved consultations with "
    "{stakeholders}. The department noted that further details would be "
    "published in the coming weeks.",

    "A study published in {journal} on {day} found that {finding}. The "
    "researchers, based at {org}, analyzed {n} cases over a period of "
    "{years} years. The study's lead author said the results 'suggest a "
    "modest but measurable effect,' and called for further peer-reviewed "
    "research before drawing firm conclusions.",

    "{org} announced quarterly results on {day}, reporting revenue of "
    "{amount} for the {period}, compared with {amount2} a year earlier. "
    "The company attributed the change to {reason}. Shares moved "
    "{move} in after-hours trading following the announcement.",

    "City officials said {infrastructure} work will begin in {region} next "
    "{month1}, as part of a {amount} plan approved by the council in "
    "{month2}. Residents can expect {impact} during construction, which is "
    "expected to conclude by {month2b}.",
]

# Short, headline-style real news — matched in length/format to
# FAKE_HEADLINE_TEMPLATES below, so the model learns to tell fabricated
# claims from real ones by CONTENT, not just by length or punctuation style.
REAL_HEADLINE_TEMPLATES = [
    "{org} reports {metric} rose {pct}% in {period}",
    "{org} confirms {policy} will take effect in {month1}",
    "Study from {org} finds {finding}",
    "{org} announces {period} revenue of {amount}",
    "City approves {amount} plan for {infrastructure} in {region}",
    "{org} spokesperson addresses {policy} timeline in {month1} briefing",
    "Researchers at {org} publish findings on {finding} in {journal}",
    "{org2} analysts note {metric} trend in {region} report",
]

FAKE_TEMPLATES = [
    "SHOCKING: {org} DOESN'T want you to know this ONE simple trick that "
    "{claim}!!! Doctors are FURIOUS after this {region} mom discovered the "
    "secret that Big {industry} has been hiding for YEARS. Click NOW before "
    "this gets taken down!!!",

    "BREAKING: Anonymous insider reveals {org} secretly {claim}. Sources "
    "close to the matter, who wish to remain unnamed, say the cover-up goes "
    "all the way to the top. Mainstream media REFUSES to report on this. "
    "Share before they DELETE it!!!",

    "You won't BELIEVE what happens when you {action}! This {region} "
    "grandmother tried it and the results left everyone SPEECHLESS. "
    "Experts hate her because she found this one weird trick that "
    "{claim}. Number 3 will shock you.",

    "URGENT WARNING: {org} is putting {substance} in {product} and NO ONE "
    "is talking about it! Wake up, sheeple! The government doesn't want "
    "this information getting out because it would destroy the entire "
    "{industry} industry overnight. Forward this to everyone you know!!!",

    "EXPOSED: The REAL reason {org} banned {topic} has FINALLY leaked, and "
    "it's not what they told you. A whistleblower leaked documents proving "
    "the whole thing was staged. This changes EVERYTHING. They don't want "
    "you to see this before it's too late!",
]

# Short, calmly-worded headline-style fake claims — no exclamation marks or
# caps-lock, so the model learns to catch fabricated/conspiratorial CONTENT
# even when the surface style looks sober rather than sensational.
FAKE_HEADLINE_TEMPLATES = [
    "Scientist: {conspiracy_claim}",
    "Leaked document reveals {org} plans to {conspiracy_action}",
    "Whistleblower says {org} has been secretly {conspiracy_gerund} for years",
    "New report claims {substance} found in {product} nationwide",
    "Insider confirms {org} covered up {conspiracy_claim_lower}",
    "Official memo shows {org} intends to make {conspiracy_action} mandatory",
    "{region} doctor warns that {product} is causing {vague_ailment}",
    "Government document proves {org} tracks citizens via {substance}",
]


orgs = ["the Department of Labor", "Reuters", "the Federal Reserve", "Stanford University",
        "the World Health Organization", "Acme Corp", "the city council", "MIT",
        "the Ministry of Health", "the National Weather Service"]
orgs2 = ["Morgan Stanley", "Goldman Sachs", "an independent think tank", "Bloomberg Economics"]
metrics = ["unemployment", "inflation", "consumer spending", "housing starts", "exports"]
periods = ["first quarter", "second quarter", "past month", "past year"]
regions = ["the Midwest", "California", "the eurozone", "Southeast Asia", "the local area", "rural Ohio"]
months = ["January", "March", "May", "July", "September", "November"]
journals = ["The Lancet", "Nature", "the Journal of Public Health", "Science"]
findings = ["a modest link between sleep duration and productivity",
            "no significant difference between the two treatment groups",
            "a small increase in recovery rates among the treatment group"]
industries = ["Pharma", "Sugar", "Oil", "Tech", "Banking"]
claims = ["cures everything overnight", "is hiding the cure for aging",
          "wants to control your mind", "will make you rich instantly",
          "is secretly harming your children"]
actions = ["mix baking soda with this common kitchen item",
           "stop eating this one vegetable", "do this every morning"]
substances = ["microchips", "mind-control chemicals", "toxic additives"]
products = ["your tap water", "your food supply", "children's vaccines"]
topics = ["this supplement", "this documentary", "this book"]

conspiracy_claims = [
    "the human subcutaneous microchip will be mandatory for everyone",
    "5G towers are secretly harvesting personal data for foreign governments",
    "the water supply is being deliberately altered to affect fertility rates",
    "a new law will require citizens to carry a digital tracking implant",
    "the flu vaccine contains a hidden nanotechnology tracking device",
    "commercial airplanes are secretly spraying mind-altering chemicals",
    "a global authority is planning to ban all privately owned currency",
]
conspiracy_actions = [
    "implant tracking chips in all newborns",
    "replace paper currency with mandatory digital identification",
    "monitor private conversations through household appliances",
    "restrict travel for anyone who has not registered biometric data",
]
conspiracy_gerunds = [
    "monitoring citizens through smart devices", "altering historical records",
    "suppressing a cure for a common illness", "manipulating food supply chains",
]
vague_ailments = ["unexplained fatigue", "memory loss", "chronic headaches", "mood changes"]


def fill_fake_headline(t):
    return t.format(
        org=random.choice(orgs), region=random.choice(regions),
        substance=random.choice(substances), product=random.choice(products),
        conspiracy_claim=random.choice(conspiracy_claims),
        conspiracy_claim_lower=random.choice(conspiracy_claims),
        conspiracy_action=random.choice(conspiracy_actions),
        conspiracy_gerund=random.choice(conspiracy_gerunds),
        vague_ailment=random.choice(vague_ailments),
    )


def fill_real_headline(t):
    return t.format(
        org=random.choice(orgs), org2=random.choice(orgs2),
        metric=random.choice(metrics), pct=round(random.uniform(0.1, 4.5), 1),
        period=random.choice(periods), policy="new zoning rules",
        month1=random.choice(months), finding=random.choice(findings),
        amount=f"${random.randint(1,900)}M", infrastructure=random.choice(
            ["road resurfacing", "water main repair", "bridge maintenance"]),
        region=random.choice(regions), journal=random.choice(journals),
    )

def fill(t):
    return t.format(
        org=random.choice(orgs), org2=random.choice(orgs2), day=f"{random.choice(months)} {random.randint(1,28)}",
        metric=random.choice(metrics), pct=round(random.uniform(0.1, 4.5), 1),
        period=random.choice(periods), caveat="seasonal effects may have played a role",
        n=random.randint(200, 5000), region=random.choice(regions),
        month1=random.choice(months), month2=random.choice(months), month2b=random.choice(months),
        policy="the new zoning rules", stakeholders="local businesses and residents",
        journal=random.choice(journals), finding=random.choice(findings),
        years=random.randint(2, 10), amount=f"${random.randint(1,900)}M",
        amount2=f"${random.randint(1,900)}M", reason="softer demand in overseas markets",
        move=random.choice(["up slightly", "down slightly", "little changed"]),
        infrastructure=random.choice(["road resurfacing", "water main", "bridge repair"]),
        impact="minor traffic delays", industry=random.choice(industries),
        claim=random.choice(claims), action=random.choice(actions),
        substance=random.choice(substances), product=random.choice(products),
        topic=random.choice(topics),
    )

def make_row(is_fake, headline_style=False):
    if headline_style:
        template = random.choice(FAKE_HEADLINE_TEMPLATES if is_fake else REAL_HEADLINE_TEMPLATES)
        text = fill_fake_headline(template) if is_fake else fill_real_headline(template)
    else:
        template = random.choice(FAKE_TEMPLATES if is_fake else REAL_TEMPLATES)
        text = fill(template)
    if is_fake:
        source = random.choice(["thetruthexposed.biz", "patriot-newsnow.info", "viral-alert24.com",
                                 "real-news-network.co", "unfiltered-daily.net"])
    else:
        source = random.choice(["reuters.com", "apnews.com", "bbc.com", "npr.org",
                                 "nature.com", "bloomberg.com"])
    return text, int(is_fake), source

def generate(n_per_class=250, headline_fraction=0.35, out_path="data/sample_dataset.csv"):
    n_headline = int(n_per_class * headline_fraction)
    n_paragraph = n_per_class - n_headline
    rows = (
        [make_row(False, headline_style=False) for _ in range(n_paragraph)] +
        [make_row(False, headline_style=True) for _ in range(n_headline)] +
        [make_row(True, headline_style=False) for _ in range(n_paragraph)] +
        [make_row(True, headline_style=True) for _ in range(n_headline)]
    )
    random.shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source"])  # label: 1 = fake, 0 = real
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path} ({n_headline*2} headline-style, {n_paragraph*2} paragraph-style)")

if __name__ == "__main__":
    generate()
