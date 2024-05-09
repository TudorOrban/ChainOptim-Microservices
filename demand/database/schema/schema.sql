-- MySQL dump 10.13  Distrib 8.2.0, for Win64 (x86_64)
--
-- Host: localhost    Database: chainoptim-demand-db
-- ------------------------------------------------------
-- Server version	8.2.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;


--
-- Table structure for table `client_orders`
--

DROP TABLE IF EXISTS `client_orders`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `client_orders` (
  `id` int NOT NULL AUTO_INCREMENT,
  `client_id` int NOT NULL,
  `raw_material_id` int DEFAULT NULL,
  `component_id` int DEFAULT NULL,
  `quantity` double DEFAULT NULL,
  `order_date` timestamp NULL DEFAULT NULL,
  `estimated_delivery_date` timestamp NULL DEFAULT NULL,
  `delivery_date` timestamp NULL DEFAULT NULL,
  `organization_id` int NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `delivery_warehouse_id` int DEFAULT NULL,
  `delivery_factory_id` int DEFAULT NULL,
  `company_id` varchar(255) DEFAULT NULL,
  `delivered_quantity` double DEFAULT NULL,
  `status` enum('INITIATED','NEGOTIATED','PLACED','DELIVERED','CANCELED') DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `client_id` (`client_id`),
  KEY `raw_material_id` (`raw_material_id`),
  KEY `component_id` (`component_id`),
  KEY `organization_id` (`organization_id`),
  KEY `client_warehouse_id` (`delivery_warehouse_id`),
  KEY `client_delively_factory_id` (`delivery_factory_id`),
  CONSTRAINT `client_orders_ibfk_1` FOREIGN KEY (`client_id`) REFERENCES `clients` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=42 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `client_orders`
--

LOCK TABLES `client_orders` WRITE;
/*!40000 ALTER TABLE `client_orders` DISABLE KEYS */;
INSERT INTO `client_orders` VALUES (1,1,NULL,1,NULL,NULL,NULL,NULL,1,'2024-03-06 20:03:10','2024-04-26 14:09:39',NULL,NULL,'13251',NULL,'DELIVERED'),(2,3,NULL,1,1123,'2022-03-12 08:05:53','2022-03-18 08:05:51','2022-03-19 08:05:53',1,'2024-03-27 14:04:55','2024-04-17 18:40:32',NULL,NULL,'#5323',1123,'DELIVERED'),(3,4,NULL,3,5315,'2022-04-12 08:05:53','2022-03-23 08:05:53','2022-03-24 13:31:21',1,'2024-03-27 14:12:45','2024-04-17 18:40:32',NULL,NULL,'#123',5300,'DELIVERED'),(4,3,NULL,4,233,'2022-05-12 08:05:53','2022-05-13 08:05:53','2022-05-13 08:30:33',1,'2024-03-27 13:58:51','2024-04-17 18:40:32',NULL,NULL,'#142',23,'DELIVERED'),(5,3,NULL,3,2312225,'2022-06-01 08:05:53','2022-06-28 08:05:54','2022-07-02 08:05:53',1,'2024-03-27 13:58:52','2024-04-17 18:40:32',NULL,NULL,'#123',51000,'DELIVERED'),(6,3,NULL,4,3434,NULL,NULL,NULL,1,'2024-04-03 11:56:48','2024-04-17 18:40:32',NULL,NULL,NULL,NULL,'DELIVERED'),(7,3,NULL,5,3,NULL,NULL,NULL,1,'2024-04-03 11:57:08','2024-04-06 12:06:47',NULL,NULL,NULL,NULL,NULL),(8,3,NULL,1,153,NULL,NULL,NULL,1,'2024-04-03 14:14:20','2024-04-17 18:40:32',NULL,NULL,NULL,NULL,'CANCELED'),(9,3,NULL,1,352.012,'2022-07-01 08:05:53','2022-07-09 08:05:53','2022-07-10 08:05:53',1,'2024-04-03 14:14:20','2024-04-17 18:40:32',NULL,NULL,NULL,351,'DELIVERED'),(10,3,NULL,1,32,NULL,NULL,NULL,1,'2024-04-04 12:43:55','2024-04-06 12:01:52',NULL,NULL,NULL,NULL,NULL),(11,3,NULL,1,1536,NULL,NULL,NULL,1,'2024-04-04 12:43:55','2024-04-06 12:01:52',NULL,NULL,NULL,NULL,NULL),(12,3,NULL,1,463,NULL,NULL,NULL,1,'2024-04-04 12:43:55','2024-04-17 18:40:32',NULL,NULL,NULL,NULL,'CANCELED'),(13,3,NULL,1,643,NULL,NULL,NULL,1,'2024-04-04 12:43:55','2024-04-06 12:01:52',NULL,NULL,NULL,NULL,NULL),(14,3,NULL,1,3464,NULL,NULL,NULL,1,'2024-04-04 12:43:55','2024-04-06 12:01:52',NULL,NULL,NULL,NULL,NULL),(15,3,NULL,1,4,NULL,NULL,NULL,1,'2024-04-04 12:43:55','2024-04-17 18:40:56',NULL,NULL,NULL,NULL,'DELIVERED'),(16,3,NULL,1,326,'2022-11-12 08:05:53','2022-11-16 08:05:53','2022-11-19 08:05:53',1,'2024-04-04 12:43:55','2024-04-17 18:40:56',NULL,NULL,NULL,325,'DELIVERED'),(17,3,NULL,1,124,NULL,NULL,NULL,1,'2024-04-04 14:34:49','2024-04-17 18:40:32',NULL,NULL,NULL,325,'DELIVERED'),(18,3,NULL,1,644,'2023-02-12 08:05:53','2023-03-01 08:05:53','2023-02-27 08:05:53',1,'2024-04-04 14:34:49','2024-04-17 18:40:32',NULL,NULL,NULL,640,'DELIVERED'),(19,3,NULL,2,352,NULL,NULL,NULL,1,'2024-04-04 14:34:49','2024-04-17 18:40:32',NULL,NULL,NULL,NULL,'DELIVERED'),(23,3,NULL,4,64,'2023-01-12 08:05:53','2023-01-15 08:05:53','2023-01-14 08:05:53',1,'2024-04-04 14:34:49','2024-04-17 18:40:32',NULL,NULL,NULL,54,'NEGOTIATED'),(24,3,NULL,1,12,NULL,NULL,NULL,1,'2024-04-04 14:36:09','2024-04-06 12:01:13',NULL,NULL,NULL,NULL,NULL),(25,3,NULL,7,1235,'2023-07-19 08:05:53','2023-08-01 08:05:53','2023-08-01 05:03:43',1,'2024-04-04 14:36:10','2024-04-06 12:22:12',NULL,NULL,NULL,1235,NULL),(26,3,NULL,2,24,NULL,NULL,NULL,1,'2024-04-04 14:36:10','2024-04-06 12:22:12',NULL,NULL,NULL,1235,NULL),(27,3,NULL,5,153,NULL,NULL,NULL,1,'2024-04-04 14:36:10','2024-04-06 12:01:13',NULL,NULL,NULL,NULL,NULL),(28,3,NULL,1,64,NULL,NULL,NULL,1,'2024-04-04 14:36:10','2024-04-06 12:01:13',NULL,NULL,NULL,NULL,NULL),(29,3,NULL,1,365,NULL,NULL,NULL,1,'2024-04-04 14:36:10','2024-04-06 12:01:13',NULL,NULL,NULL,NULL,NULL),(30,3,NULL,1,214,NULL,NULL,NULL,1,'2024-04-04 14:36:10','2024-04-06 12:01:13',NULL,NULL,NULL,NULL,NULL),(31,3,NULL,1,NULL,NULL,NULL,NULL,1,'2024-04-16 15:58:25','2024-04-16 15:58:25',NULL,NULL,NULL,NULL,NULL),(32,3,NULL,1,NULL,NULL,NULL,NULL,1,'2024-04-16 18:59:40','2024-04-16 18:59:40',NULL,NULL,NULL,NULL,NULL),(33,3,NULL,1,NULL,NULL,NULL,NULL,1,'2024-04-17 08:52:46','2024-04-17 08:52:46',NULL,NULL,NULL,NULL,NULL),(34,4,NULL,1,NULL,NULL,NULL,NULL,1,'2024-04-17 12:33:10','2024-04-17 12:33:10',NULL,NULL,NULL,NULL,NULL),(35,3,NULL,4,13213,NULL,NULL,NULL,1,'2024-04-17 12:38:03','2024-04-17 12:38:03',NULL,NULL,'#132',NULL,NULL),(36,3,NULL,4,124,NULL,NULL,NULL,1,'2024-04-17 13:10:32','2024-04-26 14:11:09',NULL,NULL,'12215121',NULL,'PLACED'),(37,4,NULL,1,NULL,NULL,NULL,NULL,1,'2024-04-17 15:19:31','2024-04-17 15:19:31',NULL,NULL,NULL,NULL,NULL),(38,4,NULL,1,4031,NULL,'2024-04-17 21:04:54',NULL,1,'2024-04-17 16:04:54','2024-04-17 19:51:30',NULL,NULL,NULL,NULL,'CANCELED'),(39,3,NULL,2,NULL,NULL,NULL,NULL,1,'2024-04-26 08:23:10','2024-04-26 08:35:43',NULL,NULL,'1321',NULL,'PLACED'),(40,3,NULL,3,NULL,NULL,NULL,NULL,1,'2024-04-26 08:23:10','2024-04-26 08:23:10',NULL,NULL,'N/',NULL,'PLACED'),(41,3,NULL,3,NULL,NULL,NULL,NULL,1,'2024-04-26 08:40:16','2024-04-26 08:40:16',NULL,NULL,'123',NULL,'NEGOTIATED');
/*!40000 ALTER TABLE `client_orders` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `client_performances`
--

DROP TABLE IF EXISTS `client_performances`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `client_performances` (
  `id` int NOT NULL AUTO_INCREMENT,
  `client_id` int NOT NULL,
  `client_performance_report` json DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `client_id` (`client_id`),
  CONSTRAINT `client_performances_ibfk_1` FOREIGN KEY (`client_id`) REFERENCES `clients` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `client_performances`
--

LOCK TABLES `client_performances` WRITE;
/*!40000 ALTER TABLE `client_performances` DISABLE KEYS */;
INSERT INTO `client_performances` VALUES (4,3,'{\"totalDelays\": 6.0, \"averageDelayPerOrder\": 0.75, \"totalDeliveredOrders\": 8, \"componentPerformances\": {\"1\": {\"componentId\": 1, \"componentName\": \"Chip\", \"firstDeliveryDate\": [2022, 3, 19, 8, 5, 53], \"averageOrderQuantity\": 493.6, \"totalDeliveredOrders\": 5.0, \"totalDeliveredQuantity\": 2468.0, \"averageShipmentQuantity\": 0.0, \"averageDeliveredQuantity\": 492.4, \"deliveredPerOrderedRatio\": 492.4, \"deliveredQuantityOverTime\": {\"0.0\": 1123.0, \"55.0\": 23.0, \"113.0\": 351.0, \"245.0\": 325.0, \"345.0\": 640.0}}, \"2\": {\"componentId\": 2, \"componentName\": \"Test Component 2\", \"firstDeliveryDate\": [2022, 7, 2, 8, 5, 53], \"averageOrderQuantity\": 51235.0, \"totalDeliveredOrders\": 1.0, \"totalDeliveredQuantity\": 51235.0, \"averageShipmentQuantity\": 0.0, \"averageDeliveredQuantity\": 51000.0, \"deliveredPerOrderedRatio\": 51000.0, \"deliveredQuantityOverTime\": {\"0.0\": 51000.0}}, \"4\": {\"componentId\": 4, \"componentName\": \"Camera\", \"firstDeliveryDate\": [2023, 1, 14, 8, 5, 53], \"averageOrderQuantity\": 64.0, \"totalDeliveredOrders\": 1.0, \"totalDeliveredQuantity\": 64.0, \"averageShipmentQuantity\": 0.0, \"averageDeliveredQuantity\": 54.0, \"deliveredPerOrderedRatio\": 54.0, \"deliveredQuantityOverTime\": {\"0.0\": 54.0}}, \"7\": {\"componentId\": 7, \"componentName\": \"New Component\", \"firstDeliveryDate\": [2023, 8, 1, 5, 3, 43], \"averageOrderQuantity\": 1235.0, \"totalDeliveredOrders\": 1.0, \"totalDeliveredQuantity\": 1235.0, \"averageShipmentQuantity\": 0.0, \"averageDeliveredQuantity\": 1235.0, \"deliveredPerOrderedRatio\": 1235.0, \"deliveredQuantityOverTime\": {\"0.0\": 1235.0}}}, \"averageTimeToShipOrder\": 10.5, \"averageDelayPerShipment\": 0.75, \"averageShipmentsPerOrder\": 0.0, \"ratioOfOnTimeOrderDeliveries\": 0.25, \"ratioOfOnTimeShipmentDeliveries\": 0.0}','2024-04-06 13:52:35','2024-04-06 21:05:21');
/*!40000 ALTER TABLE `client_performances` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `client_shipments`
--

DROP TABLE IF EXISTS `client_shipments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `client_shipments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `client_order_id` int NOT NULL,
  `quantity` float DEFAULT NULL,
  `shipment_starting_date` timestamp NULL DEFAULT NULL,
  `estimated_arrival_date` timestamp NULL DEFAULT NULL,
  `arrival_date` timestamp NULL DEFAULT NULL,
  `transporter_type` varchar(255) DEFAULT NULL,
  `status` varchar(255) DEFAULT NULL,
  `source_location_id` int DEFAULT NULL,
  `destination_location_id` int DEFAULT NULL,
  `destination_location_type` varchar(255) DEFAULT NULL,
  `current_location_latitude` float DEFAULT NULL,
  `current_location_longitude` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `source_location_id` (`source_location_id`),
  KEY `destination_location_id` (`destination_location_id`),
  KEY `client_order_id` (`client_order_id`),
  CONSTRAINT `client_shipments_ibfk_1` FOREIGN KEY (`client_order_id`) REFERENCES `client_orders` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `client_shipments`
--

LOCK TABLES `client_shipments` WRITE;
/*!40000 ALTER TABLE `client_shipments` DISABLE KEYS */;
INSERT INTO `client_shipments` VALUES (1,1,120,NULL,NULL,NULL,'Ship','Initiated',1,6,NULL,NULL,NULL),(2,1,12000,NULL,NULL,NULL,'Ship','Initiated',1,6,NULL,NULL,NULL),(5,32,31321,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(6,32,31321,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(7,32,313212000,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(8,32,31321,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(9,32,313212000,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(10,32,31321,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(11,32,313212000,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
/*!40000 ALTER TABLE `client_shipments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `clients`
--

DROP TABLE IF EXISTS `clients`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `clients` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) DEFAULT NULL,
  `organization_id` int NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `location_id` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `FKcdasbtce0ulnp7hhdryn3mo5s` (`organization_id`),
  KEY `FK62w6tt0p4ti7f1ofonportcsg` (`location_id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `clients`
--

LOCK TABLES `clients` WRITE;
/*!40000 ALTER TABLE `clients` DISABLE KEYS */;
INSERT INTO `clients` VALUES (1,'LA Client 1',2,'2024-03-04 21:07:43','2024-03-04 23:07:43',NULL),(2,'Motors Internationals',1,'2024-03-06 09:34:45','2024-03-23 16:20:45',13),(3,'AKio Motors',1,'2024-03-06 09:35:10','2024-03-06 11:35:10',6),(4,'Lu bu Co.',1,'2024-03-06 09:35:35','2024-03-06 11:35:35',5),(7,'Ford',1,'2024-03-21 18:49:38','2024-03-21 20:49:38',3),(8,'Ford',1,'2024-03-21 18:50:17','2024-03-21 20:50:17',10);
/*!40000 ALTER TABLE `clients` ENABLE KEYS */;
UNLOCK TABLES;

/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-05-01 15:35:46
