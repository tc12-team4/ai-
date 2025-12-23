# database/results_db.py
"""
Base de données pour stocker les résultats IA
Critère 11 (7%) : Tables structurées + index optimisés
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ResultsDB:
    """
    Base de données SQLite pour stocker tous les résultats du pipeline
    
    Tables :
    - tickets : Informations principales
    - rag_docs : Documents récupérés par RAG
    - responses : Réponses générées
    - escalations : Cas escaladés
    """
    
    def __init__(self, db_path: str = "./doxa_results.db"):
        """
        Initialise la connexion et crée les tables si nécessaire
        
        Args:
            db_path: Chemin vers le fichier SQLite
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Pour accéder par nom de colonne
        
        self._create_tables()
        self._create_indexes()
        
        print(f"✅ Base de données initialisée : {self.db_path}")
    
    def _create_tables(self):
        """Crée les tables si elles n'existent pas"""
        
        cursor = self.conn.cursor()
        
        # Table principale : tickets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                
                -- Triage
                coherent INTEGER,
                type_question TEXT,
                intention TEXT,
                exceptions TEXT,  -- JSON array
                
                -- Query Analysis
                resume TEXT,
                keywords TEXT,  -- JSON array
                categorie TEXT,
                reformulation TEXT,
                
                -- Evaluation
                confidence REAL,
                decision TEXT,
                escalade_reason TEXT,
                priorite_escalade TEXT,
                
                -- Timing
                execution_time_total REAL,
                execution_time_triage REAL,
                execution_time_query REAL,
                execution_time_retrieval REAL,
                execution_time_response REAL
            )
        """)
        
        # Table : rag_docs (documents récupérés)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_id INTEGER,
                score REAL,
                rank INTEGER,
                text_preview TEXT,
                
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
            )
        """)
        
        # Table : responses (réponses générées)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                response_text TEXT NOT NULL,
                langue TEXT,
                quality_score REAL,
                word_count INTEGER,
                has_structure INTEGER,
                generated_at TEXT,
                
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
            )
        """)
        
        # Table : escalations (cas escaladés)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                escalade_reason TEXT NOT NULL,
                priorite TEXT NOT NULL,
                escalade_at TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                resolved_at TEXT,
                
                FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
            )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Crée les index pour optimiser les requêtes"""
        
        cursor = self.conn.cursor()
        
        # Index sur tickets
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_categorie ON tickets(categorie)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_timestamp ON tickets(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_decision ON tickets(decision)")
        
        # Index sur rag_docs
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_ticket ON rag_docs(ticket_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_docs(source)")
        
        # Index sur responses
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_responses_ticket ON responses(ticket_id)")
        
        # Index sur escalations
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_escalations_ticket ON escalations(ticket_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_escalations_resolved ON escalations(resolved)")
        
        self.conn.commit()
    
    def save_ticket(self, result: dict):
        """
        Sauvegarde un ticket complet avec tous ses résultats
        
        Args:
            result: Dictionnaire résultat de l'orchestrator
        """
        
        cursor = self.conn.cursor()
        
        # Extraire les données
        ticket_id = result['ticket_id']
        status = result['status']
        
        triage = result.get('triage', {})
        query_data = result.get('query_data', {})
        evaluation = result.get('evaluation', {})
        exec_time = result.get('execution_time', {})
        
        # Préparer les valeurs
        values = (
            ticket_id,
            result.get('question', ''),
            datetime.now().isoformat(),
            status,
            
            # Triage
            1 if triage.get('coherent') else 0,
            triage.get('type_question', ''),
            triage.get('intention', ''),
            json.dumps(triage.get('exceptions', [])),
            
            # Query Analysis
            query_data.get('resume', ''),
            json.dumps(query_data.get('mots_cles', [])),
            query_data.get('categorie', ''),
            query_data.get('reformulation', ''),
            
            # Evaluation
            evaluation.get('confidence_finale', 0.0),
            evaluation.get('decision', ''),
            result.get('escalade_reason', ''),
            evaluation.get('priorite_escalade', ''),
            
            # Timing
            self._parse_time(exec_time.get('total', '0s')),
            self._parse_time(exec_time.get('triage', '0s')),
            self._parse_time(exec_time.get('query_processing', '0s')),
            self._parse_time(exec_time.get('retrieval', '0s')),
            self._parse_time(exec_time.get('response', '0s'))
        )
        
        # Insérer ticket
        cursor.execute("""
            INSERT OR REPLACE INTO tickets VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, values)
        
        # Sauvegarder les documents RAG
        if 'retrieval' in result:
            self._save_rag_docs(ticket_id, result['retrieval'])
        
        # Sauvegarder la réponse générée
        if 'response_data' in result:
            self._save_response(ticket_id, result['response_data'])
        
        # Sauvegarder l'escalade si applicable
        if status == 'escalated':
            self._save_escalation(ticket_id, result)
        
        self.conn.commit()
        print(f"💾 Ticket {ticket_id} sauvegardé dans la base")
    
    def _save_rag_docs(self, ticket_id: str, retrieval: dict):
        """Sauvegarde les documents RAG récupérés"""
        
        cursor = self.conn.cursor()
        
        for i, chunk in enumerate(retrieval.get('chunks', []), 1):
            cursor.execute("""
                INSERT INTO rag_docs (ticket_id, source, chunk_id, score, rank, text_preview)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticket_id,
                chunk.get('source', ''),
                chunk.get('chunk_id', 0),
                chunk.get('score', 0.0),
                i,
                chunk.get('text', '')[:200]  # Preview 200 chars
            ))
    
    def _save_response(self, ticket_id: str, response_data: dict):
        """Sauvegarde la réponse générée"""
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO responses (
                ticket_id, response_text, langue, quality_score, 
                word_count, has_structure, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            response_data.get('response', ''),
            response_data.get('langue', ''),
            response_data.get('quality_score', 0.0),
            response_data.get('word_count', 0),
            1 if response_data.get('has_structure') else 0,
            datetime.now().isoformat()
        ))
    
    def _save_escalation(self, ticket_id: str, result: dict):
        """Sauvegarde un cas d'escalade"""
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO escalations (
                ticket_id, escalade_reason, priorite, escalade_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            ticket_id,
            result.get('reason', result.get('escalade_reason', '')),
            result.get('priority', result.get('priorite_escalade', 'moyenne')),
            datetime.now().isoformat()
        ))
    
    def _parse_time(self, time_str: str) -> float:
        """Convertit '5.23s' en float 5.23"""
        try:
            return float(time_str.replace('s', ''))
        except:
            return 0.0
    
    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """Récupère un ticket par son ID"""
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_statistics(self) -> Dict:
        """Calcule des statistiques sur tous les tickets"""
        
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total tickets
        cursor.execute("SELECT COUNT(*) FROM tickets")
        stats['total_tickets'] = cursor.fetchone()[0]
        
        # Par status
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM tickets 
            GROUP BY status
        """)
        stats['by_status'] = dict(cursor.fetchall())
        
        # Par catégorie
        cursor.execute("""
            SELECT categorie, COUNT(*) 
            FROM tickets 
            WHERE categorie IS NOT NULL
            GROUP BY categorie
            ORDER BY COUNT(*) DESC
        """)
        stats['by_categorie'] = dict(cursor.fetchall())
        
        # Temps moyen
        cursor.execute("""
            SELECT AVG(execution_time_total) 
            FROM tickets 
            WHERE execution_time_total > 0
        """)
        stats['avg_execution_time'] = cursor.fetchone()[0] or 0.0
        
        # Confiance moyenne
        cursor.execute("""
            SELECT AVG(confidence) 
            FROM tickets 
            WHERE confidence > 0
        """)
        stats['avg_confidence'] = cursor.fetchone()[0] or 0.0
        
        # Escalades
        cursor.execute("SELECT COUNT(*) FROM escalations WHERE resolved = 0")
        stats['escalations_pending'] = cursor.fetchone()[0]
        
        return stats
    
    def get_recent_tickets(self, limit: int = 10) -> List[Dict]:
        """Récupère les tickets les plus récents"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM tickets 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Ferme la connexion"""
        self.conn.close()


# Fonction utilitaire
def create_database(db_path: str = "./doxa_results.db") -> ResultsDB:
    """Crée ou ouvre la base de données"""
    return ResultsDB(db_path)


if __name__ == "__main__":
    # Test de la base de données
    db = create_database("./test_doxa.db")
    
    # Test : Sauvegarder un ticket
    test_ticket = {
        "ticket_id": "T001",
        "question": "Comment activer 2FA ?",
        "status": "retrieved",
        "triage": {
            "coherent": True,
            "type_question": "question",
            "intention": "Activer 2FA",
            "exceptions": []
        },
        "query_data": {
            "resume": "Client veut activer 2FA",
            "mots_cles": ["2FA", "sécurité", "authentification"],
            "categorie": "securite",
            "reformulation": "Comment activer l'authentification 2FA ?"
        },
        "evaluation": {
            "decision": "traiter",
            "confidence_finale": 0.85
        },
        "execution_time": {
            "total": "5.23s",
            "triage": "2.1s",
            "query_processing": "1.8s",
            "retrieval": "1.33s"
        }
    }
    
    db.save_ticket(test_ticket)
    
    # Récupérer stats
    stats = db.get_statistics()
    print("\n📊 Statistiques :")
    print(json.dumps(stats, indent=2))
    
    db.close()