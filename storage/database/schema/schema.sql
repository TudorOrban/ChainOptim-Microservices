-- MySQL dump 10.13  Distrib 8.2.0, for Win64 (x86_64)
--
-- Host: localhost    Database: chainoptim-supply-db
-- ------------------------------------------------------
-- Server version	8.2.0


--
-- Table structure for table `compartments`
--

DROP TABLE IF EXISTS `compartments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `compartments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `warehouse_id` int NOT NULL,
  `organization_id` int NOT NULL,
  `data_json` json DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `warehouse_id` (`warehouse_id`),
  KEY `organization_id` (`organization_id`),
  CONSTRAINT `compartments_ibfk_1` FOREIGN KEY (`warehouse_id`) REFERENCES `warehouses` (`id`),
  CONSTRAINT `compartments_ibfk_2` FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `compartments`
--

LOCK TABLES `compartments` WRITE;
/*!40000 ALTER TABLE `compartments` DISABLE KEYS */;
INSERT INTO `compartments` VALUES (1,'Comp 1',1,1,'{}'),(2,'Compr 21',1,1,'{\"crateSpecs\": [{\"crateId\": 1, \"maxCrates\": 16.0, \"componentId\": 1}, {\"crateId\": 2, \"maxCrates\": 100.0, \"componentId\": 2}], \"currentCrates\": [{\"crateId\": 1, \"numberOfCrates\": 10}]}'),(3,'Compr 21',1,1,'{\"crateSpecs\": [{\"crateId\": 1, \"maxCrates\": 12.0, \"componentId\": 1}, {\"crateId\": 2, \"maxCrates\": 60.0, \"componentId\": 2}], \"currentCrates\": [{\"crateId\": 1, \"numberOfCrates\": 8}, {\"crateId\": 2, \"numberOfCrates\": 58}]}'),(4,'C 14',1,1,'{\"crateSpecs\": [{\"crateId\": 1, \"maxCrates\": 129.0, \"componentId\": 1}, {\"crateId\": 2, \"maxCrates\": 41.0, \"componentId\": 2}], \"currentCrates\": [{\"crateId\": 1, \"numberOfCrates\": 20}, {\"crateId\": 2, \"numberOfCrates\": 40}]}'),(5,'NewComp',1,1,'null'),(6,'NC2',1,1,'null'),(7,'weqw',1,1,'null'),(8,'NCC@!',1,1,'null'),(9,'NC12',1,1,'{\"crateSpecs\": [{\"crateId\": 1, \"maxCrates\": 2132.0, \"componentId\": null}, {\"crateId\": 2, \"maxCrates\": 21.23, \"componentId\": null}], \"currentCrates\": null}'),(10,'BDSA1',1,1,'{\"crateSpecs\": [{\"crateId\": 1, \"maxCrates\": 121.0, \"componentId\": null}, {\"crateId\": 2, \"maxCrates\": 213.21, \"componentId\": null}], \"currentCrates\": null}'),(11,'New!',1,1,'{\"crateSpecs\": [{\"crateId\": 1, \"maxCrates\": 231.0, \"componentId\": null}, {\"crateId\": 2, \"maxCrates\": 21421.0, \"componentId\": null}, {\"crateId\": 1, \"maxCrates\": 1231.0, \"componentId\": null}, {\"crateId\": 2, \"maxCrates\": 1535.0, \"componentId\": null}], \"currentCrates\": null}');
/*!40000 ALTER TABLE `compartments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `crates`
--

DROP TABLE IF EXISTS `crates`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `crates` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `component_id` int NOT NULL,
  `quantity` decimal(10,0) NOT NULL,
  `volume_m3` decimal(10,0) DEFAULT NULL,
  `stackable` tinyint(1) DEFAULT NULL,
  `height_m` decimal(10,0) DEFAULT NULL,
  `organization_id` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `component_id` (`component_id`),
  KEY `organization_id` (`organization_id`),
  CONSTRAINT `crates_ibfk_1` FOREIGN KEY (`component_id`) REFERENCES `components` (`id`),
  CONSTRAINT `crates_ibfk_2` FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `crates`
--

LOCK TABLES `crates` WRITE;
/*!40000 ALTER TABLE `crates` DISABLE KEYS */;
INSERT INTO `crates` VALUES (1,'CRate 1',1,1,123,1,3,1),(2,'Cargo Crate',2,4,19,1,2,1);
/*!40000 ALTER TABLE `crates` ENABLE KEYS */;
UNLOCK TABLES;


--
-- Table structure for table `warehouse_inventory_items`
--

DROP TABLE IF EXISTS `warehouse_inventory_items`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `warehouse_inventory_items` (
  `id` int NOT NULL AUTO_INCREMENT,
  `warehouse_id` int NOT NULL,
  `raw_material_id` int DEFAULT NULL,
  `component_id` int DEFAULT NULL,
  `product_id` int DEFAULT NULL,
  `quantity` float DEFAULT NULL,
  `minimum_required_quantity` float DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `organization_id` int DEFAULT NULL,
  `company_id` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `raw_material_id` (`raw_material_id`),
  KEY `component_id` (`component_id`),
  KEY `product_id` (`product_id`),
  KEY `warehouse_id` (`warehouse_id`),
  KEY `organization_id` (`organization_id`),
  CONSTRAINT `warehouse_inventory_items_ibfk_1` FOREIGN KEY (`warehouse_id`) REFERENCES `factories` (`id`),
  CONSTRAINT `warehouse_inventory_items_ibfk_2` FOREIGN KEY (`raw_material_id`) REFERENCES `raw_materials` (`id`),
  CONSTRAINT `warehouse_inventory_items_ibfk_3` FOREIGN KEY (`component_id`) REFERENCES `components` (`id`),
  CONSTRAINT `warehouse_inventory_items_ibfk_4` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`),
  CONSTRAINT `warehouse_inventory_items_ibfk_5` FOREIGN KEY (`warehouse_id`) REFERENCES `warehouses` (`id`),
  CONSTRAINT `warehouse_inventory_items_ibfk_6` FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `warehouse_inventory_items`
--

LOCK TABLES `warehouse_inventory_items` WRITE;
/*!40000 ALTER TABLE `warehouse_inventory_items` DISABLE KEYS */;
INSERT INTO `warehouse_inventory_items` VALUES (1,6,NULL,1,1,341,240,'2024-03-12 16:38:07','2024-04-30 09:40:51',1,NULL),(2,6,NULL,4,1,30,240,'2024-03-12 16:38:07','2024-04-30 09:40:51',1,NULL),(3,6,NULL,1,1,214,NULL,'2024-04-03 14:37:44','2024-08-09 19:35:30',1,NULL),(4,6,NULL,1,1,400,NULL,'2024-04-03 14:37:44','2024-08-09 19:35:30',1,NULL),(5,6,NULL,1,1,214,NULL,'2024-04-03 14:38:51','2024-08-02 11:42:12',1,NULL),(6,6,NULL,1,1,400,NULL,'2024-04-03 14:38:51','2024-08-02 11:42:12',1,NULL),(7,6,NULL,1,1,214,NULL,'2024-04-03 14:39:24','2024-08-02 11:42:12',1,NULL),(8,2,NULL,1,1,400,NULL,'2024-04-03 14:39:24','2024-08-09 19:29:48',1,NULL),(9,6,NULL,1,1,214,NULL,'2024-04-03 14:39:26','2024-08-02 11:42:12',1,NULL),(10,6,NULL,1,1,400,NULL,'2024-04-03 14:39:26','2024-08-02 11:42:12',1,NULL),(11,6,NULL,1,1,330,NULL,'2024-04-03 14:39:28','2024-08-02 11:42:12',1,NULL),(12,6,NULL,1,21,400,NULL,'2024-04-03 14:39:28','2024-08-02 11:42:12',1,NULL),(13,6,NULL,1,1,350,200,'2024-04-03 14:39:29','2024-08-02 11:42:12',1,NULL),(14,6,NULL,1,21,400,NULL,'2024-04-03 14:39:29','2024-08-02 11:42:12',1,NULL),(15,6,NULL,2,3,200,300,'2024-04-30 03:48:12','2024-08-02 11:42:12',1,NULL),(16,4,NULL,5,11,3112,NULL,'2024-08-09 13:30:08','2024-08-09 19:35:30',1,NULL),(17,5,NULL,8,14,123,NULL,'2024-08-09 13:33:59','2024-08-09 19:35:30',1,NULL);
/*!40000 ALTER TABLE `warehouse_inventory_items` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `warehouses`
--

DROP TABLE IF EXISTS `warehouses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `warehouses` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) DEFAULT NULL,
  `location_id` int DEFAULT NULL,
  `organization_id` int NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `location_id` (`location_id`),
  KEY `FKs5ri0quwqfiti3scdftfij40q` (`organization_id`),
  CONSTRAINT `FKs5ri0quwqfiti3scdftfij40q` FOREIGN KEY (`organization_id`) REFERENCES `organizations` (`id`),
  CONSTRAINT `warehouses_ibfk_1` FOREIGN KEY (`location_id`) REFERENCES `locations` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `warehouses`
--

LOCK TABLES `warehouses` WRITE;
/*!40000 ALTER TABLE `warehouses` DISABLE KEYS */;
INSERT INTO `warehouses` VALUES (1,'NJ Warehouse',1,1,'2024-03-04 21:07:37','2024-08-03 11:45:06'),(2,'Chicago warehouse 2',14,1,'2024-03-06 09:34:29','2024-03-23 16:40:53'),(4,'Shenzen Warehouse',22,1,'2024-03-06 09:51:05','2024-08-05 11:16:28'),(5,'Kyoto Warehouse',6,1,'2024-03-06 09:51:15','2024-03-06 11:51:15'),(6,'LA Warehouse',3,1,'2024-03-06 09:51:24','2024-03-06 11:51:24'),(9,'Warehouse S.E.A.',5,1,'2024-03-21 17:56:30','2024-04-22 21:26:16'),(10,'QWwq',5,1,'2024-08-08 10:31:09','2024-08-08 13:31:09'),(12,'Warehouse S.E.A.',38,1,'2024-08-08 11:03:22','2024-08-08 14:03:22');
/*!40000 ALTER TABLE `warehouses` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-08-27 22:56:30