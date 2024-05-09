package org.chainoptimnotifications.shared.email.service;

public interface EmailService {

    void sendEmail(String to, String subject, String text);
}
