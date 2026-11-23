# TripMates Lite Web End-to-End Functional Evaluation

Date: 2026-11-23 (Asia/Kuala_Lumpur)

## Scope and environment

This is one observed browser session of the Streamlit prototype, not a user
usability, accessibility, deployment-load, or statistical study. The app ran
locally at `http://127.0.0.1:8507` with Python 3.11.9 and Streamlit 1.63.0 on
Windows. The browser was the Codex in-app browser. Test history and weather
cache were isolated with `TRIPMATES_HISTORY_FILE` and
`TRIPMATES_WEATHER_CACHE_FILE`; the normal user history database was not used.

## Input

- Date: 2026-11-23; origin: POI 1, Dataran Merdeka.
- Time window: 09:00-17:00; compulsory admission budget: RM160.
- No additional must-visit or preferred attraction supplied.
- Planning conditions: `S1 Heavy Rain`, nominal trigger; speed: `Quick preview`.
- The fixed S1 choice controlled the replanning scenario. The sidebar displayed
  live weather, but live weather did not select the test scenario.

## Observed browser workflow

| Step | Expected result | Observation | Status |
| --- | --- | --- | --- |
| Create itinerary | Initial route renders | Initial itinerary rendered with 9 stops, route map, timeline, finish 16:44, and RM25 admission cost | Pass |
| Trigger replanning | S1 disruption produces a repaired suffix | The app reported Heavy Rain at 11:00 during an in-progress visit; a revised itinerary rendered with 7 stops, route map, revised timeline, finish 16:47, RM105 cost, WES 447.33, and a 767.39 ms displayed replanning time | Pass |
| Save | Confirmation appears and a record is persisted | Saved `E2E 2026-11-23 S1 Quick Preview`; the isolated SQLite database contained one `tripmates_lite` record with scenario `S1` | Pass |
| Reopen | Saved map, itinerary, inputs, and results are shown | The saved-itineraries view displayed the named record, map, 7-stop timeline, trip setup, WES 447.33, and the same 767.39 ms replanning result | Pass |

No application error appeared in this workflow. The observed screen used one
browser and one viewport. This evaluation supports the statement that these
four UI functions worked in the tested local configuration; it does not support
claims about user ease of use, other scenarios, cross-browser behavior, or
production reliability.

## Reproduction notes

Set the two `TRIPMATES_*_FILE` environment variables to disposable paths, run
`python -m streamlit run app.py`, and follow the input and workflow above. The
isolated test database and cached weather file are local QA artefacts, not
released study data. The application source in `app.py` defines the UI and the
optional storage-path overrides.
