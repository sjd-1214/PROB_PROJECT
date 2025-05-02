# Spotify Tracks Dataset Information

The application expects a CSV file named `spotify_tracks.csv` in the project root directory with the following structure:

## Expected Columns

### Core Features
- **name**: Track name
- **artists**: Artist name(s)
- **popularity**: A value between 0-100 representing the track's popularity
- **release_date**: The date the track was released

### Audio Features
- **danceability**: How suitable a track is for dancing (0.0 to 1.0)
- **energy**: A measure representing intensity and activity (0.0 to 1.0)
- **key**: The key the track is in (integers map to pitches using standard Pitch Class notation)
- **loudness**: Overall loudness in decibels (dB), typically between -60 and 0 db
- **mode**: Modality of the track (1 = major, 0 = minor)
- **speechiness**: Presence of spoken words (higher values indicate more speech-like recordings)
- **acousticness**: A confidence measure of whether the track is acoustic (1.0) or not (0.0)
- **instrumentalness**: Predicts whether a track contains no vocals (1.0) or contains vocals (0.0)
- **liveness**: Detects presence of an audience (higher values represent higher probability of live recording)
- **valence**: Musical positiveness (0.0 = negative/sad, 1.0 = positive/happy)
- **tempo**: Overall estimated tempo in beats per minute (BPM)
- **time_signature**: Estimated time signature (3, 4, etc.)
- **duration_ms**: Track length in milliseconds

## Sample Data Format

| name | artists | popularity | release_date | danceability | energy | key | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence | tempo | time_signature | duration_ms |
|------|---------|------------|--------------|--------------|--------|-----|----------|------|-------------|--------------|------------------|----------|---------|-------|----------------|------------|
| Track 1 | Artist A | 65 | 2020-01-15 | 0.72 | 0.65 | 5 | -6.3 | 0 | 0.05 | 0.12 | 0.0 | 0.08 | 0.43 | 126.8 | 4 | 215640 |
| Track 2 | Artist B | 78 | 2019-08-22 | 0.81 | 0.88 | 1 | -4.2 | 1 | 0.08 | 0.24 | 0.0 | 0.11 | 0.65 | 118.3 | 4 | 193250 |

## Data Sources

If you don't have this dataset, you can:

1. Use the Spotify API to collect track data (requires a Spotify Developer account)
2. Find similar datasets on platforms like Kaggle
3. Generate a synthetic dataset based on the column specifications above

## Recommended Dataset Size

For optimal performance, the dataset should contain at least 1,000 tracks to enable meaningful statistical analysis.
