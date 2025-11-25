// Constraints and indexes for RAG graph
// Node labels: Document, Section, Entity, Table, Page

// Documents
CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Sections
CREATE INDEX section_doc_page IF NOT EXISTS
FOR (s:Section) ON (s.document_id, s.page);

// Entities
CREATE CONSTRAINT entity_key IF NOT EXISTS
FOR (e:Entity) REQUIRE e.key IS UNIQUE;

// Tables
CREATE INDEX table_doc_page IF NOT EXISTS
FOR (t:Table) ON (t.document_id, t.page);

// Pages
CREATE INDEX page_doc_num IF NOT EXISTS
FOR (p:Page) ON (p.document_id, p.number);

