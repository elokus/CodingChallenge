# Teil 1: Semantische Suche
PDF-Dokumente: Dir werden PDF-Dokumente mit definiertem Inhalt bereitgestellt.
Such-Tool Entwicklung: Erstelle ein Stück Software, in dem du Suchanfragen von Nutzern eingeben kannst. Es reicht dabei, wenn das im Code direkt passiert, es muss keine UI o.ä. entwickelt werden. Es ist ebenfalls kein Interface für das Hochladen oder Auswählen der PDFs nötig, auf diese kann einfach per Dateisystem an fixer Stelle zugegriffen werden.
Semantische Suche: Identifiziere relevante Abschnitte in den PDF-Dokumenten basierend auf den Benutzeranfragen mittels semantischer Suche. Es reicht dabei, wenn die produzierten Daten nur flüchtig vorgehalten werden, es muss nicht zwingend persistiert werden.
Antwort-Generierung: Generiere und präsentiere die Antworten mit Hilfe der Ergebnisse der semantischen Suche und des LLM. Es reicht auch hierbei ein einfacher Output auf stdout o.ä..

## Lösung:
In script main.py wird die semantische Suche implementiert und nochmals im Notebook CodingChallenge.ipynb dargestellt.

## Anmerkung:
Da wir hier viel mit tabellarischen Daten arbeiten, wäre hier ein multimodaler Ansatz nützlich. Also neben den extrahierten Texten auch die Tabellen und PDF seiten als Bilddaten zu speichern und mit der semantischen Suche zu verknüpfen. Alternativ kann man die Multimodalen Fähigkeiten von GPT-4 nutzen um aus den Bildern die Tabellen Daten zu extrahieren und in einer "object_map" an ein DataFrame zu verknüpfen.

---

# Teil 2: Skalierung
Funktioniert deine Lösung auch für 500 PDFs? Was gibt es bei so vielen PDFs zu beachten? Wo sind bottlenecks oder etwaige Probleme zu erwarten?
Wie würde man es lösen, wenn der Nutzer fragt: ‘Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 3000 Stunden’?

## Antwort Skalierbarkeit:

In der Lösung zu Teil 1 wurde eine lokaler VectorStore benutzt der die Daten im Arbeitsspeicher hält. Dies ist nicht skalierbar, da der Arbeitsspeicher begrenzt ist.

Um die Skalierbarkeit zu gewährleisten würde man hier auf einen Anbieter wie Pinecone zurückgreifen. Dafür habe ich der Initialisierungs Funktion ein Argument hinzugefügt, dass es ermöglicht einen anderen VectorStore zu initialisieren.

### Bottlenecks:

Beim indexieren der Dokumente und beim erstellen der Vektoren:
- Memory: Die Datenmenge kann nicht im Arbeitsspeicher gehalten werden
- CPU: Die Berechnung der Vektoren kann sehr rechenintensiv sein
- Rate Limit bei embedding Erstellung seitens OpenAI

Beim Suchen:
- Irrlevante Ergebnisse: Die semantische Suche kann auch irrelevante Ergebnisse liefern
- Performance: Die Suche kann sehr langsam sein sofern die Datenmenge auf lokale Hardware beschränkt ist

## Teil 2.1: Metadata Filter

Um die Anfrage "Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 3000 Stunden" zu bearbeiten, wäre gerade im Kontext hoher Anzahl an Dokumenten ein Filter auf Metadaten sinnvoll. Alternativ könnte man die selbe Frage auch im Postprocessing Schritt filtern. Jedoch wäre das nicht gerade kosteneffizient.
Pinecone bietet die Möglichkeit direkt über operatoren "$gt", "$lt", "$eq", etc. zu filtern.
In der Lösung zu Teil 1 mit lokalem llama_index VectorStoreIndex ist zurzeit nur exakte Suche möglich.

Ich habe dafür einen Workaround implementiert unter scripts/metadata_filtering.py
Neben der reinen filterung muss auch die Anfrage per LLM auf mögliche Metadaten Filter geprüft werden.
Dazu ist ebenfalls eine Lösung implementiert unter scripts/metadata_filtering.py mit der Nutzung von Langchain.

## Next Steps:
Um die Anfrage korrekt zu Beantworten, müsste man mit höherem top_k Wert die Ergebnisse der semantischen Suche nochmals im Postprocessing Schritt besser aufbereiten.

---

# Teil 3: Verifizierung
Welche Möglichkeiten gibt das Ergebnis zu evaluieren / bewerten, ohne dass das zwangsweise von einem Menschen passiert? Welche Metriken / Möglichkeiten existieren, um die Ausgabe mit der Eingabe zu vergleichen und sicherzustellen, dass die Antwort zur Eingabe passt (Gerade bei konkreten Werten, wie Lebensdauer von mehr als 3000 Stunden…)
## Antwort:
**Metrics**: Typischerweise werden für die Evaluierung von semantischen Suchen Metriken wie Correctness, Faithfullness, Relevancy. Dies geschieht mit einer Reihe von darauf optimierten Prompts. Also die Ergebnisse der RAG-Pipeline werden wieder mit einem LLM in der Rolle der Judge oder Jury evaluiert.
Dabei kann eine End-to-End Evaluierung durchgeführt werden, bei der die Anfrage, die Antwort und die Dokumente mit denen die Antwort generiert wurde, betrachtet werden.
Alternativ können auch die einzelnen Komponenten der RAG-Pipeline evaluiert werden.
So ist bei der semantischen Suche die Relevanz der gefundenen Dokumente ein wichtiger Faktor.
Bei der Antwortgenerierung ist die Qualität der Antwort ein wichtiger Faktor der sich mit Correctness und Faithfullness (nicht halluzinieren).
Das Ergebniss multipler Evaluationsanfragen kann dann mit Precision, Recall und F1-Score evaluiert werden.

**Automatisierte Evaluierung**:
- Nutzung von Testdatensets wo Ground Truth bekannt ist
- Automatisierte Generierung von Anfragen und Generierung von Ground Truth über Majortiy Voting (GPT-4)