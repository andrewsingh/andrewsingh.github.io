---
title: "Text, Not Tracks: Building Music Similarity Search Without Audio or User Data"
date: 2025-10-14T00:00:00-08:00
draft: false
math: true
ShowToc: true
TocOpen: false
---

# Try It Out
Try out the search engine here: https://songmatch.up.railway.app

| Requirements           | Features                                    |
| ---------------------- | ------------------------------------------- |
| None                   | Search for songs and view results           |
| Spotify account        | Export song results to Spotify playlists    |
| Spotify Premium acount | Stream song results directly in the web app |

# Code
**Music Similarity Search**: https://github.com/andrewsingh/semantic-song-search

**Song Search Arena**: https://github.com/andrewsingh/song-search-arena

# Introduction

I’ve been streaming music on Spotify for years. In the early days, finding music to listen to was easy, as I had very little music exposure. There was a seemingly endless amount of new good music to discover. 

As time went on and my music taste matured, I began finding it more difficult to quickly find music to play. The popular public playlists weren't a great fit for me; I tried out Spotify’s personalized playlists, but those all too often either chose songs that I’ve already been listening to a lot, or were songs that didn't match the vibe I was going for. I'd occasionally make my own playlist, but then I'd end up relying on it so often that I'd soon grow bored of it and have to make a new one.

I wanted a simple, reliable way to open a streaming app and immediately start playing music that I wanted to listen to. But what does "what I want to listen to" mean exactly? My experience often fell into one of two cases:
- I can't put the vibe I'm going for into words, but if I heard a song that matches it, I'd immediately know it. I then want to find other songs that are similar to that song, that evoke the same listening experience. This motivates the **song-to-song search** problem: given a query song, retrieve the most similar songs to the query song.
- I want to start listening to something, but can't decide on a particular song - nothing comes to mind. I want to just give a few words describing the genre and mood I'm looking for, and have a list of relevant songs appear that I can choose from. This motivates the **text-to-song search** problem: given a query string, retrieve the songs that best match the text query.

I wanted to build these two types of similarity search over my library of songs, but there was one glaring problem: I didn't have any data. I’m just an end user - I don’t have access to the files of songs that I stream, nor do I have Spotify’s vast amounts of user streaming history that could be exploited for collaborative filtering. Building something to solve my music problem seemed impossible.

At some point as I was exploring this space, a thought came to me: *what if I don’t need the songs at all?* What if I could just collect  information about the songs and artists from the internet, organize that information into structured profiles, and use these profiles as my representation of the song instead? Would this be enough to build a similarity search engine that actually works well for my two search problems?

It turns out that being constrained to represent songs through text instead of the original audio can actually be a *feature*, not a bug. Representing songs through text allows the representation to capture higher-level semantic features of the song rather than low-level audio features, which often prove more useful for real-world use cases of song-based or text-based similarity search.

### What’s Inside
This post presents a method for building a music similarity search engine purely from internet text data. The search engine performs two types of search tasks:
1. **Song-to-song** search, where the query is a song and the goal is to retrieve the most similar songs to the query song
2. **Text-to-song** search, where the query is a text string and the goal is to retrieve the songs that best match the text query.

The core idea of the method is building structured text profiles of tracks and artists, then creating dense representations of these profiles to power embedding-based similarity search. 

This post also includes a head-to-head evaluation against the current state of the art audio-text model, CLaMP 3 ([Wu et al. 2025](https://arxiv.org/abs/2502.10362)), on a limited library of music for which we have access to the original audio. Our method achieves a **72% win-rate** and a 64.5% confidence-weighted win rate against CLaMP 3, *without using any audio content or user streaming data*.

In addition, this post presents [Song Search Arena](https://github.com/andrewsingh/song-search-arena/tree/main), a lightweight web app to facilitate conducting blinded, pairwise preference evaluations between music retrieval systems. The arena is model-agnostic, supports both song queries and text queries, and automatically computes analysis metrics such as win rates and Wilson Confidence Intervals.
  
# Method
This section is divided into two major parts. The first part describes how we obtain dense semantic representations of tracks from only the track metadata, while the second part presents the scoring function for scoring and ranking candidate tracks given a search query, that uses these dense representations. 

## Input: Track Metadata
We start with a list of track metadata, obtained from the Spotify Web API. Each track object contains the following fields:

| Field            | Type     |
| ---------------- | -------- |
| Track name       | string   |
| Artist names     | string[] |
| Release date     | datetime |
| Total streams    | int      |
| Daily streams    | int      |
| Spotify track id | string   |

 Instead of using Spotify's popularity score, which we found to be unreliable, we supplement the metadata with raw lifetime and daily stream counts for each track. 

## From Metadata to Semantic Representations

Given this track metadata as input, our goal is to obtain rich semantic representations of the track that we can leverage for similarity search.

Instead of the original audio content, which we don't have access to, we use a large language model with web search to generate a **descriptive text profile** of the track. We then send each part of this profile through a text embedding model to obtain a dense representation of the profile that we can use for similarity search. 


![Metadata to embeddings pipeline](/assets/text-not-tracks/metadata_to_embeddings.svg)

### Generating Text Profiles
 For each track in our library, we use an LLM with web search to generate a structured written profile of the track. Given the track name and artist names, the LLM is instructed to search the web for relevant information about the track, then use the information to create a 6-section profile of the track. The 6 sections are
1. Genres
2. Vocal style
3. Production & sound design
4. Lyrical meaning
5. Mood & atmosphere
6. Tags

Section 1 simply lists the genres the song falls into. Sections 2-5 describe different aspects of the track: each section consists of a list of descriptive words or short phrases describing that aspect of the track. Lastly, Section 6 is a list of single-word tags to be used as a high-level overview of the track’s overall atmosphere and feel.

As an example, here is the profile for the song *Espresso* by Sabrina Carpenter:

```
Song: Espresso
Artists: Sabrina Carpenter

Genres: pop, funk-dance, synth-pop, nu-disco

Vocal Style: playful delivery, cheeky attitude, light sass, crisp diction, velvety tone, major-key brightness, melodic confidence, dynamic phrasing, minimal vocal effects, clear pop timbre

Production & Sound Design: retro synths, funky bass line, dreamlike drums, upbeat tempo, reverbed instruments, disco groove, tight melodic loops, bubblegum textures, vintage sound palette, layered rhythmic patterns

Lyrical Meaning: self-confidence boost, playful femininity power, irresistible allure, energetic addiction metaphor, carefree attitude, singing about desirability, light-hearted empowerment, flirty bravado, clever wordplay, charm as a superpower

Mood & Atmosphere: sunny upbeat energy, youthful flirtation, carefree summer vibe, playful confidence, infectious joy, feel-good escapism, breezy nonchalance, danceable lightness, warm nostalgia, party-ready excitement

Tags: catchy, confident, flirty, summery, addictive, playful, feel-good, danceable
```

Here is another example, this one for the song *Circles* by Post Malone:
```
Song: Circles
Artists: Post Malone

Genres: pop rock, indie pop, soft rock, contemporary pop

Vocal Style: laid-back delivery, warm timbre, subtle rasp, understated emotion, mellow tone, legato phrasing, mid-range vocals, catchy melodic hooks, smooth transitions, conversational style

Production & Sound Design: groove-driven drums, light acoustic guitar, dreamy synth layers, laid-back tempo, clean mixing, retro pop undertones, subtle bass line, minimalistic arrangement, touches of psychedelia, atmospheric reverb

Lyrical Meaning: relationship repetition, toxic cycle awareness, breakup inevitability, struggling to let go, emotional exhaustion, hope vs resignation, yearning for change, cold vs warmth metaphor, self-reflection, decision uncertainty

Mood & Atmosphere: wistful-yet-groovy, bittersweet nostalgia, gentle melancholy, reluctant acceptance, low-key introspective, calm resignation, subtly uplifting, relatable vulnerability, thoughtfully reflective, softly buoyant

Tags: melancholic, nostalgic, chill, reflective, catchy, groovy, contemplative, heartfelt
```

### Per-aspect Embeddings
Now that we have the complete text profiles, we need to obtain dense representations of these profiles to use in semantic similarity search. The most straightforward method would be to simply feed the entire profile into a text embedding model, and get a single embedding of the profile. However, we instead choose to **embed each aspect of the profile separately**. This design choice brings two notable benefits:
- It allows each embedding to represent an individual aspect of the track, retaining more information compared to trying to fit all aspects into a single embedding. This allows the profile to be more faithfully represented in embedding space. 
- It allows for **controllable** similarity search, where the user can tune exactly how much each aspect of the song should factor into their similarity search. This controllability personalizes the search to the user's unique notion of music similarity, and allows them to tune the search on a per-query basis. 

See [Track and Artist Similarity](#track-and-artist-similarity) for an explanation of how we compute the semantic similarity between two tracks using multiple aspect embeddings.

### Artist-level Similarity
In addition to computing the track-level similarity between a query track and candidate track, we compute the similarity between the *artists* of the respective tracks as well. The process for computing artist similarity is very similar to that of track similarity.
- We first build structured profiles of each artist in our library using our same LLM with web search: each section follows the same descriptor-list format as track profiles (aside from genres - see [Prominence-weighted Artist Genres](#prominence-weighted-artist-genres) for more details).
- We then embed each profile aspect separately, and compute the final artist-level similarity as a weighted sum of the aspect-level similarity scores.

We use only a single artist per track for artist-level similarity. If a track has multiple artists, we use the main artist as the artist for that track.

We find that the combination of track and artist together provides a more complete representation and yields stronger similarity results than just the track alone. Examples of artist profiles can be found in [Example Artist Profiles](#example-artist-profiles).
#### Prominence-weighted Artist Genres
For the genres section of an artist profile, we found that using a simple list of genres led to a poor representation of the artist, due to each genre in the list having roughly equal weight in the embedding. While this works fine at the individual track level, a genre label applied to an artist is applied to their entire library of music. It's much more likely that some genres are *more* prominent in the artist's library, while others are *less* prominent.

Therefore, we expand the genres section of the artist profile to be a list of *(genre, prominence)* pairs, where **prominence** is an integer from 1-10 indicating how prominent that genre is featured in the artist's library. We then individually embed each genre, and compute the genre similarity between two artists using a prominence-weighted cross-similarity between their genres (see [Artist Genres Similarity](#artist-genres-similarity)).
### Model Selection and API Costs
- **Web search model**: To search the internet and generate the track and artist profiles, we use Perplexity's Sonar Pro model (model name `sonar-pro` in the Perplexity API).
- **Embedding model**: To embed the text profiles and text queries, we use OpenAI's `text-embedding-3-large` model (embedding size $d = 3072$). 

| Track Profiles               |            |
| ---------------------------- | ---------- |
| Generation ($ / 1000 tracks) | $17.97     |
| Embedding ($ / 1000 tracks)  | $0.044     |
| Total ($ / 1000 tracks)      | **$18.01** |

| Artist Profiles              |           |
| ---------------------------- | --------- |
| Generation ($ / 100 artists) | $2.26     |
| Embedding ($ / 100 artists)  | $0.008    |
| Total ($ / 100 artists)      | **$2.27** |

A more detailed breakdown of the API costs for generating and embedding the track and artist profiles is given in [Detailed Cost Breakdown](#detailed-cost-breakdown).

## Retrieval and Ranking

We now have semantic representations of each track and each artist in our library, that we can use for similarity search over our library of tracks. Using these embeddings, along with supporting metadata such as stream counts and release date, we have everything we need to build our music similarity search.
### Notation
First, some definitions to make the scoring function more interpretable.
- Let $L$ be the set of **tracks** in our library
- Let $T$ be the set of track **aspects** (genres, vocal style, lyrical meaning, etc.)
- Let $A$ be the set of artist **aspects**
- For track $t \in L$, let $a_t$ be the **artist** of track $t$
- For track $t \in L$ and aspect $i \in T$, let $\textbf{e}_i(t) \in \mathbb{R}^d$ be the **embedding** of track $t$'s profile for aspect $i$
- For artist $a_t$ of track $t$ and aspect $i \in A$, let $\textbf{e}_i(a_t) \in \mathbb{R}^d$ be the **embedding** of artist $a_t$'s profile for aspect $i$
- Let $s_{T}(t)$ and $s_{D}(t)$ be the total and daily **stream counts** of track $t$ respectively
- Let $r(t)$ be the **release date** of track $t$
- For artist $a_t$ of track $t$, let $G(a_t)$ be the set of **genre items** of artist $a_t$, where each genre item $i \in G$ is a tuple $(g_i, p_i)$ of genre string $g_i$ with prominence value $p_i$

### Scoring Function for Song-to-song Search
We score a candidate track $c$ against a query track $q$, with $c, q \in L$, using the below function:
$\text{score}(q,c) = w_0 \cdot \text{track_sim}(q, c) + w_1 \cdot \text{artist_sim}(q, c) + w_2 \cdot \text{era_sim}(q, c) + w_3 \cdot \text{life_pop}(c) + w_4 \cdot \text{curr_pop}(c)$
where

{{< math >}}
\begin{aligned}
\text{track_sim}(q, c)
  &= \sum_{i \in T} \alpha_i \cdot \text{cos}(\textbf{e}_i(q), \textbf{e}_i(c)) \\[8pt]
\text{artist_sim}(q, c)
  &= \beta_{\text{genres}} \cdot \text{genres_sim}(a_q, a_c) + \sum_{\substack{i \in A,\\ i \neq \text{genres}}} \beta_i \cdot \text{cos}(\textbf{e}_i(a_q), \textbf{e}_i(a_c)) \\[8pt]
\text{era_sim}(q, c)
  &= \exp{\left( \frac{-|r(q) - r(c)|}{\gamma} \right)} \\[8pt]
\text{life_pop}(c)
  &= \frac{s_{T}(c)}{s_{T}(c) + K_{T}} \\[8pt]
\text{curr_pop}(c)
  &= \frac{s_{D}(c)}{s_{D}(c) + K_{D}} \\[8pt]
\sum_{i=0}^4 w_i &= \sum_{i \in T} \alpha_i = \sum_{i \in A} \beta_i = 1 \\[8pt]
\end{aligned}
{{< /math >}}

and

{{< math >}}
\begin{aligned}
\text{genres_sim}(a_q, a_c) 
	&= \frac{\text{cross_sim}(G(a_q), G(a_c))}{\displaystyle\sqrt{\text{cross_sim}(G(a_q), G(a_q)) + \text{cross_sim}(G(a_c), G(a_c))}} \\[8pt]
\text{cross_sim}(G(a), G(b)) &= \sum_{i \in G(a)}\sum_{j \in G(b)}p_i \cdot p_j \cdot \text{cos}(\textbf{e}(g_i), \textbf{e}(g_j)) \\[8pt]
\end{aligned}
{{< /math >}}

Note that $\text{cos}$ is used here as the cosine similarity between two vectors:

{{< math >}}
\text{cos}(\textbf{e}_1, \textbf{e}_2) = \frac{\textbf{e}_1 \cdot \textbf{e}_2}{||\textbf{e}_1|| ||\textbf{e}_2||}
{{< /math >}}
$\gamma$, $K_T$, and $K_D$ are hyperparameters that are set once, while the weights $w_i$, $\alpha_i$, and $\beta_i$ are set to defaults but are tunable by the user.

We apply this scoring formula to each candidate in our library. Once all the candidates are scored, we retrieve the top $k$ candidates by their final score and return them to the user. 

### Text-to-song Search
The formulation for text-to-song search is identical to that of song-to-song search, except for the following changes:
- For track similarity and artist similarity, the query embedding $\textbf{e}(s)$ is simply the embedding of the query string $s$. Unlike in song-to-song search, here the query embedding remains the same across different aspects: each aspect of the track and the artist is compared to the same full text query.
- The artist genres similarity is computed as a prominence-weighted sum of the genre similarities to the text query. Since there is only a single query embedding, no cross-similarity formulation is needed here.
- The era similarity component is discarded from the final score computation, since the query is simply a string $s$ with no release date.

To embed a new text query $s$, we use the same embedding model that we used to embed the track and artist profiles, to ensure the query embedding is in the same embedding space as the tracks and artists. A complete formulation of the scoring formula for text-to-song search is included in [Ranking for Text-to-song Search](#ranking-for-text-to-song-search). 

# Experiments

In this section, we evaluate how our purely **text-based** music retrieval system performs against the current state-of-the-art **audio-based** music retrieval system, using a limited private library of music for which we do have access to the original audio. We directly compare our method against CLaMP 3, a multimodal audio-text model that is the current state-of-the-art in multimodal music retrieval, in an arena-style blind evaluation. 

**Important Note**: since CLaMP 3 only uses the raw audio of the track as input, to maintain a fair evaluation, we limit our similarity search to using **only track and artist profiles** - no additional metadata. This means that our scoring function for this study only uses the track similarity and artist similarity components. This will allow us to directly compare the **representations** of the tracks in our library (text-based vs. audio-based), without additional components affecting the results. 
## Experimental Setup

We evaluate these two music retrieval systems using **pairwise preference judgments** from human raters. This approach allows us to directly measure which system better satisfies users' music discovery needs without relying on proxy metrics or assuming access to ground truth "correct" answers.

### Evaluation Methodology

**Task Structure:** Raters are presented with two side-by-side ranked lists of song recommendations, generated by different systems in response to the same query. The systems are anonymized (shown as "List A" and "List B") and their left/right position is randomized to eliminate presentation bias.

**Query Types:** We evaluate two types of queries:
- **Text queries:** Natural language descriptions of music
- **Song queries:** Seed songs used to find similar music

**Post-Processing Policy:** To ensure fair comparison, all systems' raw retrieval results are processed through a uniform policy before presentation to raters. Each system returns up to 50 candidates per query, which are then filtered to produce final recommendation lists with the following constraints:
- **List Length:** Top 5 tracks
- **Artist Diversity:** Maximum 1 track per artist
- **Track Discovery:** For song-based queries, songs by the seed songs's artist are **excluded** to focus on discovery rather than within-artist similarity.

These filters are applied identically to all systems, ensuring that differences in rater preferences reflect retrieval quality rather than post-processing strategies.

**Judgment Collection:** For each pair of lists, raters make two judgments:
1. **Preference:** Which list better satisfies the query (Left, Right, or Tie)
2. **Confidence:** How confident they are in their judgment (Low, Medium, High)

Raters use an integrated Spotify playback interface to audition tracks directly within the evaluation tool. To reduce fatigue and maintain focus, we present queries in blocks alternating between text and song-based tasks.

### Datasets
#### Library Details

|                      |             |
| -------------------- | ----------- |
| Tracks               | 5721        |
| Main artists         | 408         |
| Median total streams | 209,119,209 |
| Median daily streams | 93,738      |

![Track Release Date Histogram](/assets/text-not-tracks/library_v3.1e_release_date_hist.png)

#### Evaluation Set Details

| Genre   | Text queries | Song queries | Total queries |
| ------- | ------------ | ------------ | ------------- |
| Pop     | 4            | 9            | 13            |
| Hip-hop | 4            | 5            | 9             |
| EDM     | 4            | 4            | 8             |
| Any     | 6            | 0            | 6             |
| Total   | 18           | 18           | 36            |

**Examples of text queries:**
```
"playful carefree summery pop"
"motivational inspirational workout rap"
"laid-back chill EDM"
"beach club party"
"emotional heartbreak ballad"
```

### Analysis Methodology

We aggregate multiple judgments per query using **majority voting** to determine the winner. Each query is treated as a single independent trial, with ties excluded from win rate calculations.

**Metrics:**
- **Win Rate:** Proportion of queries where system A was preferred over system B (excluding ties)
- **95% Confidence Intervals:** Wilson score intervals for win rate estimates

We report results stratified by:
- **Overall:** All queries combined
- **Task Type:** Song queries vs. text queries separately

Additionally, we compute **confidence-weighted** results where votes are weighted by rater confidence (Low=1, Medium=2, High=3) to incorporate judgment certainty into the analysis.

## Results

#### Standard

| Query type | System A | System B | Win Rate (System A) | 95% CI Lower | 95% CI Upper | Tie Rate |
| ---------- | -------- | -------- | ------------------- | ------------ | ------------ | -------- |
| all        | ours     | CLaMP 3  | **72.00%**          | 52.42%       | 85.72%       | 30.56%   |
| song       | ours     | CLaMP 3  | 71.43%              | 43.44%       | 90.25%       | 38.89%   |
| text       | ours     | CLaMP 3  | 72.73%              | 45.35%       | 88.28%       | 22.22%   |

#### Confidence-weighted
| Query type | System A | System B | Win Rate (System A) | 95% CI Lower | 95% CI Upper | Tie Rate |
| ---------- | -------- | -------- | ------------------- | ------------ | ------------ | -------- |
| all        | ours     | CLaMP 3  | **64.52%**          | 46.95%       | 78.88%       | 13.89%   |
| song       | ours     | CLaMP 3  | 68.75%              | 44.40%       | 85.84%       | 16.67%   |
| text       | ours     | CLaMP 3  | 60.00%              | 35.75%       | 80.18%       | 11.11%   |


Our text-based music similarity search achieves a 72% standard win rate and 64.5% confidence-weighted win rate against CLaMP 3 on the music retrieval task, with a 30.5% and 13.9% tie rate respectively. This means that in the standard voting framework, our method is expected to win against CLaMP 3 50% of the time, lose 19% of the time, and tie 31% of the time. While in the confidence-weighted framework, our method is expected to win 55% of the time, lose 31% of the time, and tie 14% of the time.

Note that this is a preliminary study due to limitations of library size and number of raters. Future experiments will increase the size and diversity of the library and evaluation set, as well as include ablations of our text-based method to evaluate the effect of specific design decisions such as descriptor-based profiles and prominence-weighted artist genres. 

# Song Search Arena

To facilitate rigorous evaluation of music retrieval systems, we developed and open-sourced **Song Search Arena**, a lightweight web application for conducting blinded, pairwise preference evaluations. The tool is **model-agnostic** and can be used to compare any music retrieval systems, whether they're based on embeddings, symbolic features, hybrid approaches, or collaborative filtering.
### How It Works
Song Search Arena takes two simple inputs following standardized schemas:

**1. Queries**: A list of `EvalQuery` objects, each one a single song or text query
```json

{
	"id": "5b227d8a11f528cd6230f86f03e69fcc",
	"type": "text", // or "song"
	"text": "reflective late-night electronic",
	"genres": ["edm"],
	"track_id": null // only for song-based queries
}
```

  
**2. Retrieval Results**: A list of `EvalResponse` objects, containing each system's top-K candidates per query
```json
{
	"system_id": "songmatch_prod_v1",
	"query_id": "5b227d8a11f528cd6230f86f03e69fcc",
	"candidates": [
		{"track_id": "74tsW...", "score": 0.549, "rank": 1},
		{"track_id": "2GQEM...", "score": 0.548, "rank": 2},
		// ... top K results
		]
}

```
  
The arena handles everything else: applying uniform post-processing policies, materializing head-to-head comparisons, serving randomized tasks to raters, and collecting judgments for analysis.
### Key Features
- **Blinded Comparisons**: Systems are anonymized and randomly positioned (left/right) to eliminate bias
- **Integrated Playback**: Raters can stream tracks directly via Spotify Web Playback without leaving the evaluation interface
- **Centralized Post-Processing**: Enforces consistent filtering rules across all systems (e.g., 1-per-artist limits, seed artist exclusion for discovery mode)
- **Smart Scheduling**: Prioritizes underfilled tasks and ensures balanced coverage across queries and system pairs
- **Admin Panel**: Password-protected page for the admin to upload queries and system responses, view progress per task, and download judgments in CSV or JSON for downstream analysis
- **Analysis Script**: Script that takes in the downloaded judgments as input and computes head-to-head win rates along with statistical tests. 

The codebase is available on [GitHub](https://github.com/yourusername/song-search-arena) for anyone looking to run their own music retrieval evaluations.

# Ranking Deep Dive: Breaking Down the Scoring Function
Now that we've presented the complete formulation for scoring a candidate track, let's go through it one piece at a time to get a better understand of what each part is doing.
#### Final Score
$\text{score}(q,c) = w_0 \cdot \text{track_sim}(q, c) + w_1 \cdot \text{artist_sim}(q, c) + w_2 \cdot \text{era_sim}(q, c) + w_3 \cdot \text{life_pop}(c) + w_4 \cdot \text{curr_pop}(c)$
where $ \sum_{i=0}^4 w_i = 1$

The final score is a weighted average of five components: 
1. Track similarity
2. Artist similarity
3. Era similarity
4. Lifetime popularity
5. Current popularity

The weights $w_i$ in this average are tunable by the user to allow for controllable similarity search.
#### Track and Artist Similarity

{{< math >}}
\begin{aligned}
\text{track_sim}(q, c)
  &= \sum_{i \in T} \alpha_i \cdot \text{cos}(\textbf{e}_i(q), \textbf{e}_i(c)) \\[8pt]
\text{artist_sim}(q, c)
  &= \beta_{\text{genres}} \cdot \text{genres_sim}(a_q, a_c) + \sum_{\substack{i \in A,\\ i \neq \text{genres}}} \beta_i \cdot \text{cos}(\textbf{e}_i(a_q), \textbf{e}_i(a_c)) \\[8pt]
\end{aligned}
{{< /math >}}

where $\textbf{e}_i(t)$ is the embedding of track $t$'s profile for track aspect $i$ and $\textbf{e}_i(a_t)$ is the embedding of artist $a_t$'s profile for artist aspect $i$.

Both track similarity and artist similarity are weighted averages of aspect similarities, where each aspect similarity is calculated as the cosine similarity between the two corresponding aspect embeddings (with the exception of artist genres - see [Artist Genres Similarity](#artist-genres-similarity)).

Intuitively, given a query and candidate track, we're comparing them aspect-by-aspect, and then taking a weighted average of these aspect similarity scores. The weights in this average are also controllable by the user, allowing them to customize the search to the specific aspects they care about.

#### Era Similarity
{{< math >}}
\text{era_sim}(q, c) = \exp{\left( \frac{-|r(q) - r(c)|}{\gamma} \right)}
{{< /math >}}

Era similarity is based on the difference in time between the query and candidate's release dates, mapped to $[0, 1]$ via an exponential decay function with decay constant $\gamma$. The exponential decay function has the following properties:
- Two tracks with the exact same release date get an era similarity score of 1.
- As the time difference between their release dates increase, their era similarity score decays towards 0, with the rate of decay controlled by the the decay constant $\lambda$.
In the production system, we set $\lambda = 10950$, equal to the number of days in 30 years. With this setting, two tracks that are released 3 decades apart would have an era similarity score of 0.37. 

#### Popularity Bonuses
{{< math >}}
\begin{aligned}
	\text{life_pop}(c)
	  &= \frac{s_{T}(c)}{s_{T}(c) + K_{T}} \\[8pt]
	\text{curr_pop}(c)
	  &= \frac{s_{D}(c)}{s_{D}(c) + K_{D}} \\[8pt]
\end{aligned}
{{< /math >}}
The lifetime and current popularity bonuses are based on total and daily streams respectively, mapped to $[0, 1]$ via a saturating function with priors $K_T$ and $K_D$. This saturating function has the following properties:
- A track with $K_T$ total streams will get a lifetime popularity bonus of 0.5. Similarly, a track with $K_D$ daily streams will get a current popularity bonus of 0.5. 
- As a track's total or daily stream count increases, its respective popularity bonus approaches 1.0.
In the production system, we set $K_T = 10,000,000$ and $K_D = 10,000$. With this setting, a track with 10 million total streams receives a lifetime popularity bonus of 0.5, while a track with 10,000 daily streams receives a current popularity bonus of 0.5. 

A powerful feature of these bonuses is that by setting the weights below zero, we can turn them into *inverse* popularity bonuses, that reward tracks with fewer streams instead of more streams. Applying this choice to each of the two popularity bonuses, we have 4 possible combinations that each select for a different type of music:

| life_pop | curr_pop | Ranking outcome                                      |
| -------- | -------- | ---------------------------------------------------- |
| positive | positive | Tracks that were big historically and are big now    |
| positive | negative | Tracks that were big historically but aren't anymore |
| negative | positive | Tracks that only recently became hot                 |
| negative | negative | Tracks that are more obscure ("deep cuts")           |

#### Artist Genres Similarity

{{< math >}}
\begin{aligned}
\text{genres_sim}(a_q, a_c) 
	&= \frac{\text{cross_sim}(G(a_q), G(a_c))}{\displaystyle\sqrt{\text{cross_sim}(G(a_q), G(a_q)) + \text{cross_sim}(G(a_c), G(a_c))}} \\[8pt]
\text{cross_sim}(G(a), G(b)) &= \sum_{i \in G(a)}\sum_{j \in G(b)}p_i \cdot p_j \cdot \text{cos}(\textbf{e}(g_i), \textbf{e}(g_j)) \\[8pt]
\end{aligned}
{{< /math >}}

Unlike genre similarity between tracks, which is a single cosine similarity between genre embeddings, genre similarity for artists incorporates the *prominence* of each genre in the final similarity score. This requires separate embeddings for each individual genre and a more involved similarity computation.

We compute the overall genres similarity between two artists $a$ and $b$ as a **weighted cross-similarity** between artist $a$'s genres and artist $b$'s genres, weighted by the product of their prominence values.

One problem remains with this weighted cross-similarity: normalization. If we just normalize the cross-similarity by the sum of all the weights $p_ip_j$, then two artists with the exact same genres will have a score $< 1.0$ due to $\text{cos}(\textbf{e}(g_i), \textbf{e}(g_j)) < 1.0$ for all $g_i \neq g_j$. This means that comparing an artist to themselves, unless they only have a single genre, would result in a score $< 1.0$. 

To address this, we normalize the cross-similarity between the two artists by the square root of the sum of their self-similarities. After applying this normalization, two artists with the exact same genre-prominence pairs will receive a genres similarity score of $1.0$. 

#### Addressing Same-artist Bias
For candidate tracks that have the same artist as the query track, we have $\text{artist_sim}(q, c) = 1.0$. Since there is usually a sizeable gap between this perfect score and the highest $\text{artist_sim}$ score between two *different* artists, this phenomenon inflates tracks by the same artist in the final ranking. Moreover, it is often the case in real-world use that the goal is to discover similar songs by *different* artists, rather than similar songs by the same artist, which is much easier to find.

To mitigate this same-artist bias in the production system, for candidate tracks that are by the query artist, we replace the perfect 1.0 score with the **95th percentile** artist similarity score, where the population is the set of similarity scores between the query artist and each candidate artist. 


# Limitations
The primary limitation of this method is its reliance on information about the track existing on the internet. For very new tracks or more obscure tracks, the profile generation model may be unable to acquire sufficient information to generate the profile. In this case, it will simply set the "familiar" field to False and the remaining fields to null in its response, to protect against hallucination (see [Ensuring Data Accuracy of Track and Artist Profiles](#ensuring-data-accuracy-of-track-and-artist-profiles)).

One possible way to mitigate this issue in the future would be to rely more on the artist information for an unfamiliar track. It is much less likely for there to be insufficient information on the web about an artist than about a particular track. If the track is unfamiliar but we have a detailed profile of the artist, we can leverage that information as a prior for the track. 

Another limitation of this method is that it likely won't work as well as audio-based retrieval systems if the user is looking for tracks that are highly acoustically similar to a certain track. While representing the track through text can capture higher-level sonic characteristics of the track, it won't be able to capture the audio features in fine granularity the way an audio-based representation can. Though if we assume that in most real-world use cases, users generally want to search based on higher-level notions of similarity such as mood and atmosphere, then our text-based method performs quite well. 


# References
[1] Wu et al. ["# CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages"](https://arxiv.org/abs/2502.10362)


# Appendix

### Example Artist Profiles

#### Ariana Grande
```
Artist: Ariana Grande

Genres: Pop (10), R&B (9), Trap-pop (8), Dance-pop (7), Synth-pop (5)

Vocal Style: light lyric soprano, whistle register, breathy vocals, melismatic runs, crystal-clear tone, powerful belting, controlled vibrato, velvet smooth timbre, layered harmonies, soft falsetto, rapid dynamic shifts, emotional delivery, airy high notes, ethereal phrasing, agile transitions

Production & Sound Design: lush vocal layering, trap-influenced beats, funk-infused basslines, minimalistic synths, dreamy textures, crisp drum programming, EDM flourishes, subtle guitar elements, clean vocal mixing, percussive accents, ethereal atmospheres, hip-hop inspired rhythms, radio-friendly polish, chromatic progressions, dynamic track structures

Lyrical Themes: self-love, empowerment, romantic vulnerability, sexual independence, breakups, healing from loss, confidence, moving on, personal growth, escapism, existential reflection, relationship uncertainty, apocalyptic imagery, emotional honesty, poetic metaphors

Mood & Atmosphere: uplifting, intimate, ethereal, melancholic, sensual, reflective, hopeful, comforting, empowering, urgent, dynamic, bittersweet, sophisticated, warm, dreamlike

Cultural Context & Scene: millennial pop culture, female empowerment movement, queer-affirming spaces, internet-driven fandoms, collaborative pop networks, social media aesthetics, influence of contemporary R&B, intersection with hip-hop scene, EDM-pop crossover, vocal virtuoso tradition, Gen Z cultural iconography, fashion-conscious audience, mainstream radio sound, inclusive pop music ethos, digital-age reinvention
```

#### Kendrick Lamar
```
Artist: Kendrick Lamar

Genres: Hip-hop (10), Jazz rap (8), West Coast rap (7), Funk (6), Soul (6)

Vocal Style: conversational delivery, rapid-fire flows, dynamic voice shifts, voice modulation, storytelling cadence, spoken word inflection, multi-syllabic phrasing, emotional urgency, internal rhyming, slant rhymes, clear diction, layered vocal tracks, frequent pitch changes, nasal timbre, percussive vocal attack

Production & Sound Design: jazz-influenced beats, live instrumentation, sampled funk grooves, layered soundscapes, minimalist arrangements, syncopated drum patterns, neo-soul textures, West Coast synths, raw G-funk basslines, psychedelic flourishes, distorted textures, unexpected tempo shifts, dense musical layering, lo-fi atmospherics, avant-garde instrumentals

Lyrical Themes: racial identity, systemic oppression, self-examination, fear and anxiety, generational trauma, black empowerment, personal struggle, moral conflict, urban life realism, survivor's guilt, spiritual searching, social commentary, introspection, redemption arcs, community reflection

Mood & Atmosphere: intense introspection, restless tension, vulnerable honesty, hopeful resilience, unsettling unease, urgent conviction, existential anxiety, dark optimism, raw emotionality, somber contemplation, propulsive energy, reflective melancholy, immersive storytelling, cathartic release, brooding atmosphere

Cultural Context & Scene: progressive rap movement, Compton hip-hop roots, jazz rap revival, African-American music lineage, socially engaged lyricism, Black Lives Matter influence, mainstream/underground hybrid, Pulitzer-winning artistry, genre boundary-pushing, spoken word fusion, conscious hip-hop circles, multi-generational appeal, long-form narrative influence, album-as-art aesthetic, experimental West Coast scene
```

### Ensuring data accuracy of track and artist profiles
To mitigate hallucination and ensure accuracy of the LLM outputs, we add an explicit “familiar” boolean to the prompt output, instructing the model that if it cannot find sufficient information about the track after searching, to set this familiarity field to false and the remaining fields to null. For the track that come up as familiar = False on our first pass, we increase the search context size to high and retry those songs. For the track that still remain unfamiliar, we exclude them in our library for now. Leveraging other priors such as artist information to try and include these unfamiliar songs is left as future work.

After both medium and high search context passes, only 26 / 6189 songs in our initial library remained unfamiliar, a 99.6% success rate. Note that our initial library is primarily composed of English pop music, so the success rate is likely higher than for more varied or multilingual libraries.

### Descriptor lists instead of prose
A key design decision was to have the content of each aspect be lists of descriptors (descriptive words or short phrases), rather than written prose:

Track: Rolling in the Deep - Adele
Mood & atmosphere (Prose): 
```
"The overall atmosphere is intense, cathartic, and empowering, saturated with feelings of heartbreak, anger, and defiance. There is a palpable sense of emotional release as Adele transforms her pain into strength, inviting listeners to share in both her sorrow and her resolve. The song's energy is simultaneously uplifting and heavy, with an undercurrent of melancholy that gives way to triumphant self-assertion. It inspires a sense of solidarity with anyone who has experienced betrayal, offering a cathartic outlet for strong emotions."
```
Mood & atmosphere (Descriptors): 
```
["fiercely triumphant", "fiery defiance", "cathartic heartbreak", "vulnerable-yet-empowered", "restless determination", "stormy intensity", "righteously wounded", "stirring confidence", "tension-to-release dynamic", "drama-charged energy"]
```

 While our initial version used several sentences of prose for each aspect, we discovered that descriptor lists, while less readable to humans, were better suited as inputs to embedding models and provided more accurate representations for similarity search. We posit this is likely due to the higher **information density** of descriptor lists, whereas prose contains more "filler" that, while enhancing readability for humans, pollutes the embedding representation and adds more noise to the similarity search.


### Detailed Cost Breakdown

#### Track Profiles

| Generation                 |          |
| -------------------------- | -------- |
| Mean input tokens          | 951.34   |
| Mean output tokens         | 341.36   |
| $ / 1M input tokens        | $3.00    |
| $ / 1M output tokens       | $15.00   |
| Input cost per track       | $0.00285 |
| Output cost per track      | $0.00512 |
| Web search cost per track  | $0.01    |
| Total cost per track       | $0.01797 |
| Total cost per 1000 tracks | $17.97   |

| Embedding                  |             |
| -------------------------- | ----------- |
| Mean input tokens          | 341.36      |
| $ / 1M input tokens        | $0.13       |
| Total cost per track       | $0.00004438 |
| Total cost per 1000 tracks | $0.04438    |

Total cost (generation + embedding) per 1000 tracks: $18.0144

#### Artist Profiles

| Generation                 |          |
| -------------------------- | -------- |
| Mean input tokens          | 1192.23  |
| Mean output tokens         | 603.75   |
| $ / 1M input tokens        | $3.00    |
| $ / 1M output tokens       | $15.00   |
| Input cost per artist      | $0.00358 |
| Output cost per artist     | $0.00906 |
| Web search cost per artist | $0.01    |
| Total cost per artist      | $0.02263 |
| Total cost per 100 artists | $2.263   |

| Embedding                  |             |
| -------------------------- | ----------- |
| Mean input tokens          | 603.75      |
| $ / 1M input tokens        | $0.13       |
| Total cost per artist      | $0.00007849 |
| Total cost per 100 artists | $0.007849   |

Total cost (generation + embedding) per 100 artists: $2.2708


## Ranking for Text-to-song Search

### New notation
- For string $s$, let $\textbf{e}(s) \in \mathbb{R}^d$ be the **embedding** of $s$
### Notation (restated from [Notation](#notation))
- Let $L$ be the set of **tracks** in our library
- Let $T$ be the set of track **aspects** (genres, vocal style, lyrical meaning, etc.)
- Let $A$ be the set of artist **aspects**
- For track $t \in L$, let $a_t$ be the **artist** of track $t$
- For track $t \in L$ and aspect $i \in T$, let $\textbf{e}_i(t) \in \mathbb{R}^d$ be the **embedding** of track $t$'s profile for aspect $i$
- For artist $a_t$ of track $t$ and aspect $i \in A$, let $\textbf{e}_i(a_t) \in \mathbb{R}^d$ be the **embedding** of artist $a_t$'s profile for aspect $i$
- Let $s_{T}(t)$ and $s_{D}(t)$ be the total and daily **stream counts** of track $t$ respectively
- Let $r(t)$ be the **release date** of track $t$
- For artist $a_t$ of track $t$, let $G(a_t)$ be the set of **genre items** of artist $a_t$, where each genre item $i \in G$ is a tuple $(g_i, p_i)$ of genre string $g_i$ with prominence value $p_i$

### Scoring Function for Text-to-song Search
We score a candidate track $c$ against a query string $s$ using the below function:
$\text{score}(s,c) = w_0 \cdot \text{track_sim}(s, c) + w_1 \cdot \text{artist_sim}(s, c) + w_2 \cdot \text{life_pop}(c) + w_3 \cdot \text{curr_pop}(c)$
where
{{< math >}}
\begin{aligned}
\text{track_sim}(s, c)
  &= \sum_{i \in T} \alpha_i \cdot \text{cos}(\textbf{e}(s), \textbf{e}_i(c)) \\[8pt]
\text{artist_sim}(s, c)
  &= \beta_{\text{genres}} \cdot \frac{1}{\sum_{i \in G(a_c)} p_i} \sum_{i \in G(a_c)}p_i \cdot \text{cos}(\textbf{e}(s), \textbf{e}(g_i)) + \sum_{\substack{i \in A,\\ i \neq \text{genres}}} \beta_i \cdot \text{cos}(\textbf{e}_i(a_q), \textbf{e}_i(a_c)) \\[8pt]
\text{life_pop}(c)
  &= \frac{s_{T}(c)}{s_{T}(c) + K_{T}} \\[8pt]
\text{curr_pop}(c)
  &= \frac{s_{D}(c)}{s_{D}(c) + K_{D}} \\[8pt]
\sum_{i=0}^4 w_i &= \sum_{i \in T} \alpha_i = \sum_{i \in A} \beta_i = 1 \\[8pt]
\end{aligned}
{{< /math >}}


$K_T$, and $K_D$ are hyperparameters that are set once, while the weights $w_i$, $\alpha_i$, and $\beta_i$ are set to defaults but are tunable by the user.

We apply this scoring formula to each candidate in our library. Once all the candidates are scored, we retrieve the top $k$ candidates by their final score and return them to the user. 
