-- database/init.sql
USE beads;

-- 验证码表
CREATE TABLE IF NOT EXISTS verification_codes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(20) NOT NULL UNIQUE,
    times INT NOT NULL DEFAULT 100,
    activated_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 系统配置表（记录上次重置年月）
CREATE TABLE IF NOT EXISTS system_config (
    config_key VARCHAR(50) PRIMARY KEY,
    config_value VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 初始化配置
INSERT INTO system_config (config_key, config_value) 
VALUES ('last_reset_month', DATE_FORMAT(NOW(), '%Y-%m'));

-- 创建重置存储过程（改进版）
DELIMITER $$

CREATE PROCEDURE check_and_reset_monthly()
BEGIN
    DECLARE current_ym VARCHAR(7);
    DECLARE last_ym VARCHAR(7);
    DECLARE affected INT;
    
    SET current_ym = DATE_FORMAT(NOW(), '%Y-%m');
    
    SELECT config_value INTO last_ym 
    FROM system_config 
    WHERE config_key = 'last_reset_month';
    
    IF last_ym IS NULL OR current_ym != last_ym THEN
        UPDATE verification_codes SET times = 100;
        SET affected = ROW_COUNT();
        
        INSERT INTO system_config (config_key, config_value) 
        VALUES ('last_reset_month', current_ym)
        ON DUPLICATE KEY UPDATE config_value = current_ym;
        
        SELECT CONCAT('✅ 重置完成: ', affected, ' 条 (', COALESCE(last_ym, 'NULL'), ' → ', current_ym, ')') AS result;
    ELSE
        SELECT CONCAT('⏭️  本月已重置 (', current_ym, ')') AS result;
    END IF;
END$$

DELIMITER ;

-- Event（作为备份，但不依赖它）
SET GLOBAL event_scheduler = ON;

CREATE EVENT IF NOT EXISTS monthly_check_event
ON SCHEDULE EVERY 1 DAY
DO CALL check_and_reset_monthly();

SELECT '✅ 初始化完成' AS status;